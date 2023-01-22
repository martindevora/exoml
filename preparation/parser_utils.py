import logging
import os
import pathlib
import re
from itertools import chain
from multiprocessing import Pool

import astropy.units as u
import foldedleastsquares
import lightkurve
import numpy as np
import wotan
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.mast import Catalogs
from lcbuilder.helper import LcbuilderHelper
from lcbuilder.lcbuilder_class import LcBuilder
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder
from lcbuilder.star.HabitabilityCalculator import HabitabilityCalculator
from watson.watson import Watson


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
        create_star_inputs = create_star_inputs + [CreateStarCsvInput(str(lc_file), pixel_size, search_radius,
                                                                      object_id_regex, star_catalog, lcs_dir)]
    with Pool(processes=cores) as pool:
        pool.map(create_star_csv, create_star_inputs)

def create_star_csv(create_star_input):
    tries = 0
    lc_filename_match = re.search(create_star_input.lcs_regex, create_star_input.lc_file)
    object_id = int(lc_filename_match[2])
    mission_id = lc_filename_match[1]
    star_file = create_star_input.output_dir + '/' + mission_id + '_' + str(object_id) + '_star.csv'
    if not os.path.exists(star_file):
        while tries < 3:
            try:
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
                tries = tries + 1
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
        star_df.to_csv(star_file)
    logging.info('Processed object id ' + str(object_id))

def prepare_lc_data(file_prefix, object_id, lc_data, transit_duration, transit_period, epoch, ra, dec, star_info, long_cadence):
    cadence = 'long' if long_cadence else 'short'
    lc_data = lc_data.drop(['quality'], axis=1)
    lcbuilder = LcBuilder()
    mission, mission_prefix, id = lcbuilder.parse_object_info(object_id)
    author = lcbuilder.get_default_author(object_id, cadence)
    tpfs = lightkurve.search_targetpixelfile(object_id, exptime=cadence, author=author).download_all()
    lc_df = pd.DataFrame(columns=['#time', 'flux', 'flux_err', 'planet_model', 'eb_model', 'bckeb_model',
                                  'centroid_x', 'centroid_y', 'motion_x', 'motion_y', 'bck_flux'])
    flux_values = np.array(lc_data.flux)
    # Normalizing PDCSAP flux, but not using the normalization for detrending
    median_flux = np.nanmedian(flux_values)
    flux_values = flux_values / median_flux
    if len(np.argwhere(flux_values < 0).flatten()) > 0:
        flux_values = np.array(lc_data.flux)
        flux_values = flux_values + 2 * np.abs(np.nanmin(flux_values))
        median_flux = np.nanmedian(flux_values)
        flux_values = flux_values / median_flux
    time_values = np.array(lc_data.time)
    transit_duration = transit_duration if transit_duration < transit_period else transit_period / 2
    duration_to_period = transit_duration / transit_period
    series_len = len(time_values)
    for j in np.arange(0, 1, 1):
        logging.info("Detrending no %s for target %s with %s points with duration %s", j, object_id, series_len, transit_duration * (4 + j))
        flux_detrend_j = detrend(time_values, np.array(lc_data.flux), transit_duration * (4 + j), smooth=not long_cadence)
        lc_df['flux_' + str(j)] = flux_detrend_j
    logging.info("Detrending bck for target %s with %s points with duration %s", object_id, series_len, transit_duration * 4)
    bck_flux_detrended = np.array(lc_data.background_flux)
    bck_flux_detrend = detrend(time_values, bck_flux_detrended, transit_duration * 4, smooth=False)
    logging.info("Finished detrending for target %s with %s points", object_id, series_len)
    lc_df['bck_flux'] = bck_flux_detrend
    flux_err_values = np.array(lc_data.flux_err) / flux_values
    lc_df['#time'] = time_values
    lc_df['flux'] = flux_values
    lc_df['flux_err'] = flux_err_values
    lc_df['centroid_x'] = np.array(lc_data.centroids_x)
    lc_df['centroid_y'] = np.array(lc_data.centroids_y)
    lc_df['motion_x'] = np.array(lc_data.motion_x)
    lc_df['motion_y'] = np.array(lc_data.motion_y)
    lc_df = lc_df.fillna(0)
    lc_df.to_csv(file_prefix + '_lc_' + str(cadence) + '.csv')
    logging.info("Computing T0s for %s", object_id)
    t0s_list = LcbuilderHelper.compute_t0s(lc_data['time'].to_numpy(), transit_period, epoch, transit_duration)
    source_offsets_diffimg = []
    source_offsets_bls = []
    centroids_offsets_ra_list = []
    centroids_offsets_dec_list = []
    centroids_offsets_time_list = []
    og_time_list = []
    og_core_flux_list = []
    og_halo_flux_list = []
    logging.info("Computing TPF data for %s", object_id)
    for tpf in tpfs:
        sector_name, sector = LcbuilderHelper.mission_lightkurve_sector_extraction(mission, tpf)
        aperture = tpf.pipeline_mask
        cadence_s = np.round(np.nanmedian(tpf.time.value[1:] - tpf.time.value[:-1]) * 3600 * 24)
        cadences_per_transit = LcbuilderHelper.estimate_transit_cadences(cadence_s, transit_duration * 2)
        t0s_in_tpf_indexes = np.argwhere((t0s_list > tpf.time.value[0] - transit_duration) &
                                         (t0s_list < tpf.time.value[-1] + transit_duration)).flatten()
        if len(t0s_in_tpf_indexes) == 0:
            continue
        tpf_t0s_list = t0s_list[t0s_in_tpf_indexes]
        good_quality_t0s = []
        for t0 in tpf_t0s_list:
            t0_in_tpf_indexes = \
                np.argwhere((tpf.time.value > t0 - transit_duration) & (tpf.time.value < t0 + transit_duration)).flatten()
            cadences_ratio = len(t0_in_tpf_indexes) / cadences_per_transit
            if cadences_ratio >= 0.75:
                good_quality_t0s.append(t0)
            else:
                try:
                    tpf = tpf[(tpf.time.value < t0 - transit_duration) | (tpf.time.value > t0 + transit_duration)]
                    trash = tpf[0] #we try to make it crash and if so, we discard the TPF
                except:
                    logging.exception("TPF filtering error for %s and T0 %.02f", object_id, t0)
                    continue
        if len(good_quality_t0s) == 0:
            continue
        hdu = tpf.hdu[2].header
        wcs = WCS(hdu)
        centroids_drift = \
            Watson.compute_centroids_for_tpf(ra, dec, lc_data, tpf, wcs, transit_period, epoch, duration_to_period)
        optical_ghost_data = Watson.compute_optical_ghost_data(tpf, aperture, transit_period, epoch, duration_to_period)
        pixel_values_i = np.array(range(tpf.shape[1]))
        pixel_values_j = np.array(range(tpf.shape[2]))
        snr_map, ax = Watson.plot_pixels(tpf, title=mission_prefix + ' ' + str(id) + ' ' + sector_name + ' ' + str(sector) + ' TPF BLS Analysis',
                                         period=transit_period, epoch=epoch, duration=transit_duration, aperture_mask=aperture, dry=True)
        source_offset_px = Watson.light_centroid(snr_map, pixel_values_i, pixel_values_j)
        source_offset_bls = wcs.pixel_to_world(source_offset_px[1], source_offset_px[0])
        tpf_sub = Watson.compute_tpf_diff_image(tpf, transit_period, epoch, transit_duration)
        source_offset_diffimg_px = Watson.light_centroid(tpf_sub, pixel_values_i, pixel_values_j)
        source_offset_diffimg = wcs.pixel_to_world(source_offset_diffimg_px[1], source_offset_diffimg_px[0])
        centroids_offsets_time_list.append(centroids_drift[0])
        centroids_offsets_ra_list.append(centroids_drift[1])
        centroids_offsets_dec_list.append(centroids_drift[2])
        og_time_list.append(optical_ghost_data[0])
        og_core_flux_list.append(optical_ghost_data[1])
        og_halo_flux_list.append(optical_ghost_data[2])
        source_offsets_bls.append((source_offset_bls.ra.value, source_offset_bls.dec.value))
        source_offsets_diffimg.append((source_offset_diffimg.ra.value, source_offset_diffimg.dec.value))
    og_df = pd.DataFrame(columns=['time', 'core_flux', 'halo_flux', 'og_flux'])
    og_df['time'] = list(chain.from_iterable(og_time_list))
    og_df['core_flux'] = list(chain.from_iterable(og_core_flux_list))
    og_df['halo_flux'] = list(chain.from_iterable(og_halo_flux_list))
    og_df['og_flux'] = og_df['halo_flux'] - og_df['core_flux']
    og_df.fillna(0)
    og_df.to_csv(file_prefix + '_og_' + str(cadence) + '.csv')
    centroid_coords_df = pd.DataFrame(columns=['time', 'centroids_ra', 'centroids_dec'])
    centroid_coords_df['centroids_ra'] = list(chain.from_iterable(centroids_offsets_ra_list))
    centroid_coords_df['centroids_dec'] = list(chain.from_iterable(centroids_offsets_dec_list))
    centroid_coords_df['time'] = list(chain.from_iterable(centroids_offsets_time_list))
    centroid_coords_df.fillna(0)
    centroid_coords_df.to_csv(file_prefix + '_cent_' + str(cadence) + '.csv')
    source_offset_diffimg_ra = np.nanmedian(np.array(source_offsets_diffimg)[:, 0])
    source_offset_diffimg_dec = np.nanmedian(np.array(source_offsets_diffimg)[:, 1])
    source_offset_diffimg_ra_err = np.nanstd(np.array(source_offsets_diffimg)[:, 0])
    source_offset_diffimg_dec_err = np.nanstd(np.array(source_offsets_diffimg)[:, 1])
    source_offset_bls_ra = np.nanmedian(np.array(source_offsets_bls)[:, 0])
    source_offset_bls_dec = np.nanmedian(np.array(source_offsets_bls)[:, 1])
    source_offset_bls_ra_err = np.nanstd(np.array(source_offsets_bls)[:, 0])
    source_offset_bls_dec_err = np.nanstd(np.array(source_offsets_bls)[:, 1])
    offset_ra = np.mean([source_offset_bls_ra, source_offset_diffimg_ra])
    offset_dec = np.mean([source_offset_bls_dec, source_offset_diffimg_dec])
    offset_ra_err = np.sqrt(source_offset_bls_ra_err ** 2 + source_offset_diffimg_ra_err ** 2)
    offset_dec_err = np.sqrt(source_offset_bls_dec_err ** 2 + source_offset_diffimg_dec_err ** 2)
    offsets_df = pd.DataFrame(columns=['name', 'ra', 'dec', 'ra_err', 'dec_err'])
    offsets_df = offsets_df.append({'name': 'diff_img', 'ra': source_offset_diffimg_ra, 'dec': source_offset_diffimg_dec,
                       'ra_err': source_offset_diffimg_ra_err, 'dec_err': source_offset_diffimg_dec_err}, ignore_index=True)
    offsets_df = offsets_df.append({'name': 'px_bls', 'ra': source_offset_bls_ra, 'dec': source_offset_bls_dec,
                       'ra_err': source_offset_bls_ra_err, 'dec_err': source_offset_bls_dec_err}, ignore_index=True)
    offsets_df = offsets_df.append({'name': 'mean', 'ra': offset_ra, 'dec': offset_dec,
                       'ra_err': offset_ra_err, 'dec_err': offset_dec_err}, ignore_index=True)
    offsets_df.to_csv(file_prefix + '_offset_' + str(cadence) + '.csv')
    logging.info("Computing metrics for %s", object_id)
    lc_df['time_folded'] = foldedleastsquares.fold(lc_df['#time'].to_numpy(), transit_period, epoch + transit_period / 2)
    lc_df['time_folded_sec'] = foldedleastsquares.fold(lc_df['#time'].to_numpy(), transit_period, epoch)
    lc_df_secit = lc_df[(lc_df['time_folded_sec'] > 0.5 - duration_to_period / 2) & (lc_df['time_folded_sec'] < 0.5 + duration_to_period / 2)]
    sec_depth = 1 - lc_df_secit['flux_0'].median()
    sec_depth_err = lc_df_secit['flux_0'].std()
    lc_df_it = lc_df[(lc_df['time_folded'] > 0.5 - duration_to_period / 2) & (lc_df['time_folded'] < 0.5 + duration_to_period / 2)]
    depth = 1 - lc_df_it['flux_0'].median()
    rp = np.sqrt(depth * (star_info.radius ** 2))
    rp_rs = rp / star_info.radius
    depth_err = lc_df_it['flux_0'].std()
    semimajor_axis = \
        HabitabilityCalculator().calculate_semi_major_axis(transit_period, star_info.mass) * 215.032 #sun radii in au
    geom_albedo = sec_depth * (semimajor_axis / rp)
    albedo = 0.5
    planet_eq_temp = star_info.teff * ((star_info.radius / (2 * semimajor_axis)) ** (1 / 2)) * ((1 - albedo) ** (1 / 4))
    planet_eff_temp = star_info.teff * (sec_depth ** (1 / 4)) * (rp_rs ** (1 / 2))
    statistics_df = pd.DataFrame(columns=['name', 'value'])
    statistics_df = statistics_df.append({'name': 'depth', 'value': depth}, ignore_index=True)
    statistics_df = statistics_df.append({'name': 'sec_depth', 'value': sec_depth}, ignore_index=True)
    statistics_df = statistics_df.append({'name': 'geom_albedo', 'value': geom_albedo}, ignore_index=True)
    statistics_df = statistics_df.append({'name': 'eq_temp', 'value': planet_eq_temp}, ignore_index=True)
    statistics_df = statistics_df.append({'name': 'eff_temp', 'value': planet_eff_temp}, ignore_index=True)
    statistics_df.to_csv(file_prefix + '_metrics_' + str(cadence) + '.csv')
    #
    # lc_df['#time1'] = foldedleastsquares.fold(np.float32(time_values), transit_period, epoch + transit_period / 2)
    # lc_df = lc_df.sort_values(by=['#time1'])
    # lc_df_focus = lc_df
    # lc_df_focus = lc_df_focus[
    #     (lc_df_focus['#time1'] > 0.5 - duration_to_period * 3) & (lc_df_focus['#time1'] < 0.5 + duration_to_period * 3)]
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
    # axs.scatter(lc_df_focus['#time1'], lc_df_focus['flux_2'])
    # axs.set_ylim(0.490, 0.505)
    # plt.show()
    # plt.clf()
    # plt.close()
    return lc_df

class CreateTargetInput:
    def __init__(self, mission, id, lc_file, target_dir, models_dir, injected_files_df, cache_dir, max_lc_length=20610, force=False) -> None:
        super().__init__()
        self.mission = mission
        self.id = id
        self.lc_file = lc_file
        self.target_dir = target_dir
        self.models_dir = models_dir
        self.injected_files_df = injected_files_df
        self.max_lc_length = max_lc_length
        self.cache_dir = cache_dir
        self.force = force

def create_target_csv(create_target_input):
    mission_lightcurve_builder = MissionLightcurveBuilder()
    mission_id = create_target_input.id
    file_id = mission_id.replace(' ', '_')
    file_prefix = create_target_input.target_dir + file_id
    try:
        logging.info("Trying to get short cadence info for %s", mission_id)
        file_name = file_prefix + '_metrics_short.csv'
        if create_target_input.force or not os.path.exists(file_name):
            lcbuild_short = \
                mission_lightcurve_builder.build(MissionObjectInfo('all', mission_id, cadence='short'),
                                                 create_target_input.target_dir,
                                                 create_target_input.cache_dir)
            prepare_lc_data(file_prefix, mission_id, lcbuild_short.lc_data,
                            create_target_input.injected_files_df['duration(h)'] / 24,
                            create_target_input.injected_files_df['period'],
                            create_target_input.injected_files_df['epoch'],
                            lcbuild_short.star_info.ra, lcbuild_short.star_info.dec, lcbuild_short.star_info, False)
    except Exception as e:
        logging.exception("Failed short cadence for %s", mission_id)
    try:
        logging.info("Trying to get long cadence info for %s", mission_id)
        file_name = file_prefix + '_metrics_long.csv'
        if create_target_input.force or not os.path.exists(file_name):
            lcbuild_long = \
                mission_lightcurve_builder.build(MissionObjectInfo('all', mission_id, cadence='long'),
                                                     create_target_input.target_dir,
                                                     create_target_input.cache_dir)
            prepare_lc_data(file_prefix, mission_id, lcbuild_long.lc_data,
                                    create_target_input.injected_files_df['duration(h)'] / 24,
                                    create_target_input.injected_files_df['period'],
                                    create_target_input.injected_files_df['epoch'],
                                    lcbuild_long.star_info.ra, lcbuild_long.star_info.dec, lcbuild_long.star_info, True)
    except Exception as e:
        logging.exception("Failed long cadence for %s", mission_id)
