import gzip
import multiprocessing
import os
import pathlib
import shutil
import traceback
from multiprocessing import Pool
import astropy.units as u
import pandas as pd
import numpy as np
import wotan
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from lcbuilder.star.TicStarCatalog import TicStarCatalog
from lightkurve import TessLightCurve
from numpy.random import default_rng

from ml.physics.transit_functions import t14


def create_injection_dataframe(injections_dir, lcs_dir):
    backeb_df = pd.read_csv(injections_dir + '/ete6_backeb_data.txt', comment="#")
    eb_df = pd.read_csv(injections_dir + '/ete6_eb_data.txt', comment="#")
    planet_df = pd.read_csv(injections_dir + '/ete6_planet_data.txt', comment="#")
    df = pd.DataFrame(columns=['TIC ID', 'type', 'period', 'epoch', 'Rsec/Rpri', 'b',
                                'a/Rstar', 'duration(h)', 'depth_primary', 'depth_secondary',
                               'insolation', 'Rstar_primary', 'Rstar_secondary', 'contact_amplitude', 'Mstar'])
    i = 0
    log_count = 100
    random_number_generator = default_rng()
    for lc_file in list(pathlib.Path(lcs_dir).glob('*.fits')):
        lc = TessLightCurve.read(lc_file)
        object_id = lc.meta['OBJECT']
        object_id = int(object_id.split(' ')[1])
        leading_zeros_object_id = '{:09}'.format(object_id)
        star_file = lcs_dir + '/' + leading_zeros_object_id + '_star.csv'
        star_df = pd.read_csv(star_file)
        star_radius = star_df['radius'].iloc[1]
        try:
            star_radius = 1 if np.isnan(star_radius) else star_radius
        except:
            print("ERROR WITH " + str(star_file))
        star_mass = star_df['mass'].iloc[1]
        try:
            star_mass = 1 if np.isnan(star_mass) else star_mass
        except:
            print("ERROR WITH " + str(star_file))
        min_period = 0.5
        max_period = 10
        period = random_number_generator.uniform(min_period, max_period)
        min_lc_time = 0.5
        epoch = random_number_generator.uniform(min_lc_time, min_lc_time + period)
        duration = t14(star_radius, star_mass, period)
        duration = duration * 24
        df = df.append({'TIC ID': object_id, 'type': 'none', 'period': period,
                   'epoch': epoch,
                   'Rsec/Rpri': 0,
                   'b': 0,
                   'a/Rstar': 0, 'duration(h)': duration,
                   'depth_primary': 0,
                   'depth_secondary': 0,
                   'insolation': 0, 'Rstar_primary': 0,
                   'Rstar_secondary': 0,
                   'contact_amplitude': 0,
                   'Mstar': 0}, ignore_index=True)
        target_rows = backeb_df[backeb_df['TIC ID'] == object_id]
        for index, target_row in target_rows.iterrows():
            period = target_row['Orbital Period']
            epoch = target_row['Epoch [BTJD]']
            Rsec_Rpri = target_row['Primary Rstar'] / target_row['Secondary Rstar']
            if not np.isnan(Rsec_Rpri) and Rsec_Rpri > 0 and Rsec_Rpri < 1:
                radius_secondary = star_radius * target_row['Rsec/Rpri']
                duration = t14(star_radius, star_mass, period, R_p=radius_secondary)
            elif not np.isnan(target_row['Secondary Star Depth']) and target_row['Secondary Star Depth'] > 0:
                radius_secondary = np.sqrt((target_row['Secondary Star Depth'] / 1e6) * (star_radius ** 2))
                duration = t14(star_radius, star_mass, period, R_p=radius_secondary)
            else:
                duration = t14(star_radius, star_mass, period)
            duration = duration * 24
            df = df.append({'TIC ID': target_row['TIC ID'], 'type': 'bckEB', 'period': period,
                       'epoch': epoch,
                       'Rsec/Rpri': target_row['Secondary Rstar'] / target_row['Primary Rstar'],
                       'b': target_row['Impact Parameter'],
                       'a/Rstar': 0, 'duration(h)': duration,
                       'depth_primary': target_row['Primary Star Depth'],
                       'depth_secondary': target_row['Primary Star Depth'],
                       'insolation': 0, 'Rstar_primary': target_row['Primary Rstar'],
                       'Rstar_secondary': target_row['Secondary Rstar'],
                       'contact_amplitude': target_row['Contact Binary Amplitude'],
                       'Mstar': 0}, ignore_index=True)
        target_rows = eb_df[eb_df['TIC ID'] == object_id]
        for index, target_row in target_rows.iterrows():
            period = target_row['Orbital Period']
            epoch = target_row['Epoch [BTJD]']
            Rsec_Rpri = target_row['Primary Rstar'] / target_row['Secondary Rstar']
            if not np.isnan(Rsec_Rpri) and Rsec_Rpri > 0 and Rsec_Rpri < 1:
                radius_secondary = star_radius * target_row['Rsec/Rpri']
                duration = t14(star_radius, star_mass, period, R_p=radius_secondary)
            elif not np.isnan(target_row['Secondary Star Depth']) and target_row['Secondary Star Depth'] > 0:
                radius_secondary = np.sqrt((target_row['Secondary Star Depth'] / 1e6) * (star_radius ** 2))
                duration = t14(star_radius, star_mass, period, R_p=radius_secondary)
            else:
                duration = t14(star_radius, star_mass, period)
            duration = duration * 24
            df = df.append({'TIC ID': target_row['TIC ID'], 'type': 'EB', 'period': period,
                       'epoch': epoch,
                       'Rsec/Rpri': target_row['Secondary Rstar'] / target_row['Primary Rstar'],
                       'b': target_row['Impact Parameter'],
                       'a/Rstar': 0, 'duration(h)': duration,
                       'depth_primary': target_row['Primary Star Depth'],
                       'depth_secondary': target_row['Primary Star Depth'],
                       'insolation': 0, 'Rstar_primary': target_row['Primary Rstar'],
                       'Rstar_secondary': target_row['Secondary Rstar'],
                       'contact_amplitude': target_row['Contact Binary Amplitude'],
                       'Mstar': 0}, ignore_index=True)
        target_rows = planet_df[planet_df['TIC ID'] == object_id]
        for index, target_row in target_rows.iterrows():
            period = target_row['Orbital Period']
            epoch = target_row['Epoch [BTJD]']
            duration = target_row['duration[h]']
            if np.isnan(duration) or duration == 0:
                Rsec_Rpri = target_row['Primary Rstar'] / target_row['Secondary Rstar']
                if not np.isnan(Rsec_Rpri) and Rsec_Rpri > 0 and Rsec_Rpri < 1:
                    radius_secondary = star_radius * target_row['Rsec/Rpri']
                    duration = t14(star_radius, star_mass, period, R_p=radius_secondary)
                elif not np.isnan(target_row['depth_secondary']) and target_row['depth_secondary'] > 0:
                    radius_secondary = np.sqrt((target_row['depth_secondary'] / 1e6) * (star_radius ** 2))
                    duration = t14(star_radius, star_mass, period, R_p=radius_secondary)
                else:
                    duration = t14(star_radius, star_mass, period)
                duration = duration * 24
            df = df.append({'TIC ID': target_row['TIC ID'], 'type': 'planet', 'period': period,
                       'epoch': epoch,
                       'Rsec/Rpri': target_row['Rp/Rstar'],
                       'b': target_row['Impact Parameter'],
                       'a/Rstar': target_row['a/Rstar'], 'duration(h)': duration,
                       'depth_primary': target_row['depth'],
                       'depth_secondary': 0,
                       'insolation': target_row['Insolation Flux'],
                       'Rstar_primary': target_row['Rstar'],
                       'Rstar_secondary': target_row['Rstar'] * target_row['Rp/Rstar'],
                       'contact_amplitude': 0,
                       'Mstar': target_row['Mstar']}, ignore_index=True)
        if (i + 1) % log_count == 0:
            print("Processed " + str(i) + " TICs.")
            #df = df.sort_values(["TIC ID", "type"], ascending=True)
            df.to_csv(injections_dir + '/injected_objects.csv')
        i = i + 1
    df = df.sort_values(["TIC ID", "type"], ascending=True)
    df.to_csv(injections_dir + '/injected_objects.csv')

def create_star_csvs(lcs_dir, search_radius=10, pixel_size=20.25, cores=os.cpu_count() - 1):
    create_star_inputs = []
    for lc_file in list(pathlib.Path(lcs_dir).glob('*.fits')):
        create_star_inputs = create_star_inputs + [CreateStarCsvInput(lc_file, pixel_size, search_radius)]
    with Pool(processes=cores) as pool:
        pool.map(_create_star_csv, create_star_inputs)
            
def _create_star_csv(create_star_input):
    tries = 0
    while tries < 3:
        try:
            lc = TessLightCurve.read(create_star_input.lc_file)
            object_id = lc.meta['OBJECT']
            object_id = int(object_id.split(' ')[1])
            leading_zeros_object_id = '{:09}'.format(object_id)
            star_info = TicStarCatalog().catalog_info(object_id)
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
            print(e)
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
        print('Something failed when retrieving neighbours info for object id ' + str(object_id))
        traceback.print_exc()
    star_df.to_csv(lcs_dir + '/' + leading_zeros_object_id + '_star.csv')
    print('Processed object id ' + str(object_id))


class CreateStarCsvInput:
    def __init__(self, lc_file, pixel_size=20.25, search_radius=10) -> None:
        super().__init__()
        self.lc_file = lc_file
        self.search_radius = search_radius
        self.pixel_size = pixel_size * u.arcsec


def smooth_and_detrend(time_array, flux_array, detrend_window, time_threshold=0.0001, window_len=11,
                        window='blackman'):
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    flux_without_zeros = flux_array[time_array > time_threshold]
    time_without_zeros = time_array[time_array > time_threshold]
    s = np.r_[np.ones(window_len // 2), flux_without_zeros, np.ones(window_len // 2)]
    # print(len(s))
    if window_len >= 3:
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
        flux_without_zeros = np.convolve(w / w.sum(), s, mode='valid')
    flux_without_zeros = np.nan_to_num(wotan.flatten(time_without_zeros, flux_without_zeros, detrend_window,
                                                     method="biweight"))
    flux_array[time_array > time_threshold] = flux_without_zeros
    return flux_array


def create_target_csvs(lcs_dir, models_dir, cores=multiprocessing.cpu_count() - 1, max_lc_length=20610):
    injected_files_df = pd.read_csv(models_dir + '/injected_objects.csv')
    inputs = []
    for file in list(pathlib.Path(lcs_dir).glob('*.fits')):
        lc_file = str(file)
        inputs = inputs + [CreateTargetInput(lc_file, models_dir, injected_files_df, max_lc_length)]
    with Pool(processes=cores) as pool:
        pool.map(_create_target_csv, inputs)

class CreateTargetInput:
    def __init__(self, lc_file, models_dir, injected_files_df, max_lc_length=20610) -> None:
        super().__init__()
        self.lc_file = lc_file
        self.models_dir = models_dir
        self.injected_files_df = injected_files_df
        self.max_lc_length = max_lc_length

def _create_target_csv(create_target_input):
    print("Processing lc " + create_target_input.lc_file)
    lc_df = pd.DataFrame(columns=['#time', 'flux', 'flux_err', 'planet_model', 'eb_model', 'bckeb_model',
                                  'centroid_x', 'centroid_y', 'motion_x', 'motion_y', 'bck_flux'])
    try:
        lc = TessLightCurve.read(create_target_input.lc_file)
    except Exception as e:
        print("Invalid fits file " + create_target_input.lc_file)
        return
    object_id = lc.meta['OBJECT']
    object_id = int(object_id.split(' ')[1])
    injected_file_row = create_target_input.injected_files_df[create_target_input.injected_files_df['TIC ID'] == object_id]
    leading_zeros_object_id = '{:09}'.format(object_id)
    lc_length = len(lc)
    eb_model_flux = np.ones(lc_length)
    backeb_model_flux = np.ones(lc_length)
    planet_model_flux = np.ones(lc_length)
    model_file = create_target_input.models_dir + '/ebs/EBs/EBs' + '_' + leading_zeros_object_id + '.txt'
    if os.path.exists(model_file):
        model_df = pd.read_csv(model_file, header=None)
        eb_model_flux = model_df[0].to_numpy()
        eb_model_flux = eb_model_flux / np.median(eb_model_flux)
        eb_model_flux = resample_model(eb_model_flux, lc.time.value)
    model_file = create_target_input.models_dir + '/backebs/BackEBs/BackEBs_' + '_' + leading_zeros_object_id + '.txt'
    if os.path.exists(model_file):
        model_df = pd.read_csv(model_file, header=None)
        backeb_model_flux = model_df[0].to_numpy()
        backeb_model_flux = backeb_model_flux / np.median(backeb_model_flux)
        backeb_model_flux = resample_model(backeb_model_flux, lc.time.value)
        # import matplotlib.pyplot as plt
        # plt.scatter(np.linspace(lc.time.value[0], lc.time.value[-1], len(eb_model_flux)), eb_model_flux)
        # plt.scatter(lc.time.value, lc.pdcsap_flux.value / np.median(lc.pdcsap_flux.value))
        # plt.show()
    model_file = create_target_input.models_dir + '/planets/Planets/Planets' + '_' + leading_zeros_object_id + '.txt'
    if os.path.exists(model_file):
        model_df = pd.read_csv(model_file, header=None)
        planet_model_flux = model_df[0].to_numpy()
        planet_model_flux = planet_model_flux / np.median(planet_model_flux)
        planet_model_flux = resample_model(planet_model_flux, lc.time.value)
    padding_zeros_count = create_target_input.max_lc_length - lc_length
    padding_zeros_count = 0 if padding_zeros_count < 0 else padding_zeros_count
    padding_zeros_count_left = padding_zeros_count // 2 + 1 if padding_zeros_count % 2 != 0 else padding_zeros_count // 2
    padding_zeros_count_right = padding_zeros_count // 2
    padding_zeros_left = np.zeros(padding_zeros_count_left)
    padding_zeros_right = np.zeros(padding_zeros_count_right)
    padding_ones_left = np.zeros(padding_zeros_count_left)
    padding_ones_right = np.zeros(padding_zeros_count_right)
    flux_values = np.array(lc.pdcsap_flux.value)
    median_flux = np.median(flux_values)
    flux_values = flux_values / median_flux
    if len(np.argwhere(flux_values < 0).flatten()) > 0:
        flux_values = np.array(lc.pdcsap_flux.value) + 2 * np.abs(np.nanmin(lc.pdcsap_flux.value))
        median_flux = np.median(flux_values)
        flux_values = flux_values / median_flux
    time_values = lc.time.value
    duration = injected_file_row['duration(h)'].iloc[0] / 24
    duration_to_period = duration / injected_file_row['period'].iloc[0]
    for j in np.arange(0, 5, 1):
        flux_detrend_j = smooth_and_detrend(time_values, flux_values, duration_to_period * (4 + j))
        lc_df['flux_' + str(j)] = np.concatenate((padding_ones_left, flux_detrend_j, padding_ones_right))
    flux_err_values = np.array(lc.flux_err.value) / median_flux
    lc_df['#time'] = np.concatenate((padding_zeros_left, np.array(lc.time.value), padding_zeros_right))
    lc_df['flux'] = np.concatenate((padding_ones_left, flux_values, padding_ones_right))
    lc_df['flux_err'] = np.concatenate((padding_zeros_left, flux_err_values, padding_zeros_right))
    lc_df['centroid_x'] = np.concatenate((padding_zeros_left, np.array(lc.centroid_col.value), padding_zeros_right))
    lc_df['centroid_y'] = np.concatenate((padding_zeros_left, np.array(lc.centroid_row.value), padding_zeros_right))
    lc_df['motion_x'] = np.concatenate((padding_zeros_left, np.array(lc.mom_centr1.value), padding_zeros_right))
    lc_df['motion_y'] = np.concatenate((padding_zeros_left, np.array(lc.mom_centr2.value), padding_zeros_right))
    lc_df['bck_flux'] = np.concatenate((padding_zeros_left, np.array(lc.sap_bkg.value), padding_zeros_right))
    lc_df['eb_model'] = np.concatenate((padding_ones_left, np.array(eb_model_flux), padding_ones_right))
    lc_df['bckeb_model'] = np.concatenate((padding_ones_left, np.array(backeb_model_flux), padding_ones_right))
    lc_df['planet_model'] = np.concatenate((padding_ones_left, np.array(planet_model_flux), padding_ones_right))
    lc_df = lc_df.fillna(0)
    lc_df.to_csv(lcs_dir + '/' + leading_zeros_object_id + '_lc.csv')

def resample_model(model, lc_time):
    total_missing_points = len(model) - len(lc_time)
    gaps_sizes = []
    time_gaps_indexes = np.argwhere(np.abs(lc_time[1:] - lc_time[:-1]) > 0.014).flatten()
    for time_gap_index in time_gaps_indexes:
        gaps_sizes = gaps_sizes + [lc_time[time_gap_index + 1] - lc_time[time_gap_index]]
    total_gap_size = np.sum(gaps_sizes)
    gap_sizes_percent = [gap_size / total_gap_size for gap_size in gaps_sizes]
    gap_points = []
    for gap_size_percent in gap_sizes_percent[:-1]:
        gap_points = gap_points + [int(np.round(total_missing_points * gap_size_percent))]
    gap_points = gap_points + [total_missing_points - np.sum(gap_points)]
    added_gap_points = 0
    i = 0
    corrected_time_gap_indexes = np.array([])
    for time_gap_index in time_gaps_indexes:
        corrected_time_gap_index = time_gap_index + added_gap_points
        corrected_time_gap_indexes = np.concatenate((corrected_time_gap_indexes, np.arange(corrected_time_gap_index + 1, corrected_time_gap_index + 1 + gap_points[i])))
        added_gap_points = gap_points[i]
        i = i + 1
    model = np.delete(model, corrected_time_gap_indexes.astype(int))
    return model

def uncompress_data(data_dir):
    for file in os.listdir(data_dir):
        original_file_name = data_dir + '/' + file
        destination_file_name = file.split('.')[0]
        destination_file_name = data_dir + '/' + destination_file_name + '.txt'
        with gzip.open(original_file_name, 'rb') as f_in:
            with open(destination_file_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


root_dir = '/mnt/DATA-2/ete6/'
lcs_dir = root_dir + 'lcs/'
#uncompress_data('/mnt/DATA-2/ete6/backebs/BackEBs/')
#uncompress_data('/mnt/DATA-2/ete6/ebs/EBs/')
#uncompress_data('/mnt/DATA-2/ete6/planets/Planets/')
#create_star_csvs(lcs_dir)
#create_injection_dataframe(root_dir, lcs_dir)
create_target_csvs(lcs_dir, root_dir)
#_create_target_csv(CreateTargetInput(lcs_dir + '/tess2019128220341-0000000230382377-0016-s_lc.fits', root_dir, pd.read_csv(root_dir + '/injected_objects.csv')))
