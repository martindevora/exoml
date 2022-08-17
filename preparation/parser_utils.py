import logging
import os
import pathlib
import re
from multiprocessing import Pool

import astropy.units as u
import numpy as np
import wotan
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from lcbuilder.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.preparer.MissionFfiLightcurveBuilder import MissionFfiLightcurveBuilder
from lcbuilder.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder


def detrend(time_array, flux_array, detrend_window, time_threshold=0.0001, smooth=True, smooth_window_len=11,
            window='blackman'):
    flux_without_zeros = flux_array[time_array > time_threshold]
    time_without_zeros = time_array[time_array > time_threshold]
    if smooth:
        if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s = np.r_[np.ones(smooth_window_len // 2), flux_without_zeros, np.ones(smooth_window_len // 2)]
        # print(len(s))
        if smooth_window_len >= 3:
            if window == 'flat':  # moving average
                w = np.ones(smooth_window_len, 'd')
            else:
                w = eval('np.' + window + '(smooth_window_len)')
            flux_without_zeros = np.convolve(w / w.sum(), s, mode='valid')
    flux_without_zeros = np.nan_to_num(wotan.flatten(time_without_zeros, flux_without_zeros, detrend_window,
                                                     method="biweight"))
    flux_array[time_array > time_threshold] = flux_without_zeros
    return flux_array


class CreateStarCsvInput:
    def __init__(self, lc_file, pixel_size, search_radius, lcs_regex, star_catalog, output_dir) -> None:
        super().__init__()
        self.lc_file = lc_file
        self.search_radius = search_radius
        self.pixel_size = pixel_size * u.arcsec
        self.lcs_regex = lcs_regex
        self.star_catalog = star_catalog
        self.output_dir = output_dir

def create_star_csvs(lcs_dir, search_radius, pixel_size, star_catalog, fileslist_regex, object_id_regex,
                     cores=os.cpu_count() - 1):
    create_star_inputs = []
    for lc_file in list(pathlib.Path(lcs_dir).glob(fileslist_regex)):
        create_star_inputs = create_star_inputs + [CreateStarCsvInput(lc_file, pixel_size, search_radius,
                                                                      object_id_regex, star_catalog, lcs_dir)]
    with Pool(processes=cores) as pool:
        pool.map(create_star_csv, create_star_inputs)

def create_star_csv(create_star_input):
    tries = 0
    object_id = None
    leading_zeros_object_id = ''
    while tries < 3:
        try:
            lc_filename_match = re.search(create_star_input.lcs_regex, create_star_input.lc_file)
            object_id = int(lc_filename_match[1])
            leading_zeros_object_id = '{:09}'.format(object_id)
            star_info = create_star_input.star_catalog.catalog_info(object_id)
            star_df = pd.DataFrame(
                columns=['id', 'ra', 'dec', 'ld_a', 'ld_b', 'Teff', 'lum', 'logg', 'radius', 'mass', 'v', 'j', 'h',
                         'k',
                         'dist_arcsec'])
            star_df = star_df.append({'id': object_id, 'ra': star_info[11], 'dec': star_info[12],
                                      'ld_a': star_info[0][0],
                                      'ld_b': star_info[0][1], 'Teff': star_info[1],
                                      'lum': star_info[2], 'logg': star_info[3], 'radius': star_info[5],
                                      'mass': star_info[8], 'v': star_info[13], 'j': star_info[15],
                                      'h': star_info[17],
                                      'k': star_info[19], 'dist_arcsec': 0}, ignore_index=True)
            break
        except Exception as e:
            logging.exception("Failed object %s try", object_id)
    if tries >= 3:
        print("Failed downloading object id " + str(object_id))
    assert tries < 3
    try:  # Acquiring neighbours parameters
        ra = star_df['ra'].iloc[0]
        dec = star_df['dec'].iloc[0]
        if ra is not None and dec is not None:
            ticid = Catalogs.query_region(
                SkyCoord(ra, dec, unit="deg"),
                radius=create_star_input.search_radius * create_star_input.pixel_size,
                catalog="TIC"
            )[0]["ID"]
        df = Catalogs.query_object(
            "TIC" + str(ticid),
            radius=create_star_input.search_radius * create_star_input.pixel_size,
            catalog="TIC"
        )
        stars = df.to_pandas()
        sep = [0]
        pa = [0]
        c_target = SkyCoord(
            stars["ra"].values[0],
            stars["dec"].values[0],
            unit="deg"
        )
        for i in range(1, len(stars)):
            c_star = SkyCoord(
                stars["ra"].values[i],
                stars["dec"].values[i],
                unit="deg"
            )
            sep.append(
                np.round(
                    c_target.separation(c_star).to(u.arcsec).value,
                    3
                )
            )
            pa.append(
                np.round(
                    c_target.position_angle(c_star).to(u.deg).value,
                    3
                )
            )
        stars["dist_arcsec"] = sep
        stars["PA (E of N)"] = pa  # TODO should we use this?
        for index, star in stars.iterrows():
            star_df = star_df.append({'id': star['ID'], 'ra': star['ra'], 'dec': star['dec'], 'ld_a': 0,
                                      'ld_b': 0, 'Teff': star['Teff'],
                                      'lum': star['lum'], 'logg': star['logg'], 'radius': star['rad'],
                                      'mass': star['mass'], 'v': star['Vmag'], 'j': star['Jmag'],
                                      'h': star['Hmag'],
                                      'k': star['Kmag'], 'dist_arcsec': star['dist_arcsec']}, ignore_index=True)
        star_df = star_df.sort_values(["dist_arcsec"], ascending=True)
    except Exception as e:
        logging.exception('Something failed when retrieving neighbours info for object id %s', str(object_id))
    star_df.to_csv(create_star_input.output_dir + '/' + leading_zeros_object_id + '_star.csv')
    logging.info('Processed object id ' + str(object_id))

def prepare_lc_data(lc_data, transit_duration, transit_period):
    lc_df = pd.DataFrame(columns=['#time', 'flux', 'flux_err', 'planet_model', 'eb_model', 'bckeb_model',
                                  'centroid_x', 'centroid_y', 'motion_x', 'motion_y', 'bck_flux'])
    flux_values = np.array(lc_data.flux)
    median_flux = np.nanmedian(flux_values)
    flux_values = flux_values / median_flux
    if len(np.argwhere(flux_values < 0).flatten()) > 0:
        flux_values = np.array(lc_data.flux)
        flux_values = flux_values + 2 * np.abs(np.nanmin(flux_values))
        median_flux = np.nanmedian(flux_values)
        flux_values = flux_values / median_flux
    time_values = np.array(lc_data.time)
    duration_to_period = transit_duration / transit_period
    for j in np.arange(0, 5, 1):
        flux_detrend_j = detrend(time_values, flux_values, duration_to_period * (4 + j))
        lc_df['flux_' + str(j)] = flux_detrend_j
    bck_flux_detrend = detrend(time_values, flux_values, duration_to_period * 4, smooth=False)
    lc_df['bck_flux'] = bck_flux_detrend
    flux_err_values = np.array(lc_data.flux_err) / median_flux
    lc_df['#time'] = time_values
    lc_df['flux'] = flux_values
    lc_df['flux_err'] = flux_err_values
    lc_df['centroid_x'] = np.array(lc_data.centroids_x)
    lc_df['centroid_y'] = np.array(lc_data.centroids_y)
    lc_df['motion_x'] = np.array(lc_data.motion_x)
    lc_df['motion_y'] = np.array(lc_data.motion_y)
    lc_df = lc_df.fillna(0)
    return lc_df

class CreateTargetInput:
    def __init__(self, mission, id, lc_file, target_dir, models_dir, injected_files_df, cache_dir, max_lc_length=20610) -> None:
        super().__init__()
        self.mission = mission
        self.id = id
        self.lc_file = lc_file
        self.target_dir = target_dir
        self.models_dir = models_dir
        self.injected_files_df = injected_files_df
        self.max_lc_length = max_lc_length
        self.cache_dir = cache_dir

def create_target_csv(create_target_input):
    mission_lightcurve_builder = MissionLightcurveBuilder()
    mission_ffi_lightcurve_builder = MissionFfiLightcurveBuilder()
    mission_id = create_target_input.mission + ' ' + str(int(create_target_input.id))
    file_id = create_target_input.mission + '_' + str(int(create_target_input.id))
    try:
        logging.info("Trying to get short cadence info for %s", mission_id)
        file_name = create_target_input.target_dir + file_id + "_lc_short.csv"
        if not os.path.exists(file_name):
            lcbuild_short = \
                mission_lightcurve_builder.build(MissionObjectInfo(mission_id, 'all'),
                                                 create_target_input.target_dir,
                                                 create_target_input.cache_dir)
            lc_df = prepare_lc_data(lcbuild_short.lc_data,
                                    create_target_input.injected_files_df['tce_duration'] / 24,
                                    create_target_input.injected_files_df['tce_period'])
            lc_df.to_csv(file_name)
    except Exception as e:
        logging.exception("Failed short cadence for %s", mission_id)
    try:
        logging.info("Trying to get long cadence info for %s", mission_id)
        file_name = create_target_input.target_dir + file_id + "_lc_long.csv"
        if not os.path.exists(file_name):
            lcbuild_long = \
                mission_ffi_lightcurve_builder.build(MissionFfiIdObjectInfo(mission_id, 'all'),
                                                     create_target_input.target_dir,
                                                     create_target_input.cache_dir)
            lc_df = prepare_lc_data(lcbuild_long.lc_data,
                                    create_target_input.injected_files_df['tce_duration'] / 24,
                                    create_target_input.injected_files_df['tce_period'])
            lc_df.to_csv(file_name)
    except Exception as e:
        logging.exception("Failed long cadence for %s", mission_id)
