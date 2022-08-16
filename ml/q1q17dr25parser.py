import gzip
import logging
import multiprocessing
import os
import re
import sys
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
from lcbuilder.star.KicStarCatalog import KicStarCatalog
from lcbuilder.star.TicStarCatalog import TicStarCatalog
from lightkurve import TessLightCurve
from numpy.random import default_rng
from ml.physics.transit_functions import t14
from lcbuilder.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from lcbuilder.objectinfo.preparer.MissionFfiLightcurveBuilder import MissionFfiLightcurveBuilder
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder


def create_targets_df(tces_dir):
    # Merge TCEs DF and cumulative DF ignoring everything not from Kepler or K2
    # Mark TCEs as planet only if they have photometric and rv measurements
    tces_df = pd.read_csv(tces_dir + '/q1_q17_dr25_tce_2022.06.22_13.49.14.csv', comment='#')
    tces_df['tce_period'] = tces_df['tce_period'].round(2)
    cum_df = pd.read_csv(tces_dir + '/cumulative_2022.05.31_02.29.20.csv', comment='#')
    cum_df['pl_name'] = cum_df['kepler_name']
    cum_df['tce_period'] = cum_df['koi_period']
    cum_df['tce_period'] = cum_df['tce_period'].round(2)
    tces_df = pd.merge(tces_df, cum_df, on=['kepid', 'tce_period'], how='outer')
    disc_df = pd.read_csv(tces_dir + '/PSCompPars_2022.06.25_14.17.21.csv', comment='#')
    disc_df = disc_df[disc_df['disc_facility'] == 'Kepler']
    disc_df = disc_df[disc_df['discoverymethod'] == 'Transit']
    disc_df = pd.merge(tces_df, disc_df, on='pl_name', how='outer')
    disc_df['type'] = 'tce'
    disc_df[(disc_df['rv_flag'] > 0) & (disc_df['tran_flag'] > 0) & (disc_df['koi_disposition'] == 'CONFIRMED')]['type'] = 'planet'
    disc_df[(disc_df['rv_flag'] == 0) & (disc_df['tran_flag'] > 0) & (disc_df['koi_disposition'] == 'CONFIRMED')]['type'] = 'planet_transit'
    disc_df[disc_df['koi_disposition'] == 'FALSE POSITIVE']['type'] = 'fp'
    disc_df[disc_df['koi_disposition'] == 'CANDIDATE']['type'] = 'candidate'
    disc_df.to_csv(tces_dir + '/classified_tces.csv')

def prepare_labeled_lcs(csv, target_dir, cache_dir):
    tces_df = pd.read_csv(csv, comment='#')
    tces_df = tces_df[tces_df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
    failed_targets_short = []
    failed_targets_long = []
    tces_df = tces_df.sort_values(by=["kepid"], ascending=True)
    mission_lightcurve_builder = MissionLightcurveBuilder()
    mission_ffi_lightcurve_builder = MissionFfiLightcurveBuilder()
    for index, tce_row in tces_df.iterrows():
        mission_id = 'KIC ' + str(int(tce_row['kepid']))
        try:
            logging.info("Trying to get short cadence info for %s", mission_id)
            file_name = target_dir + mission_id + "_time_series_short.csv"
            if not os.path.exists(file_name):
                lcbuild_short = \
                    mission_lightcurve_builder.build(MissionObjectInfo(mission_id, 'all'), target_dir, cache_dir)
                lc_short = lcbuild_short.lc
                lcbuild_short.lc_data.to_csv(file_name)
        except Exception as e:
            logging.exception("Failed short cadence for %s", mission_id)
            failed_targets_short = failed_targets_short + [mission_id]
        try:
            logging.info("Trying to get long cadence info for %s", mission_id)
            file_name = target_dir + mission_id + "_time_series_long.csv"
            if not os.path.exists(file_name):
                lcbuild_long = \
                    mission_ffi_lightcurve_builder.build(MissionFfiIdObjectInfo(mission_id, 'all'), target_dir, cache_dir)
                sectors = lcbuild_long.sectors
                lc_long = lcbuild_long.lc
                lcbuild_long.lc_data.to_csv(file_name)
            #logging.info("Downloading neighbour stars for %s" + mission_id)
            #TODO get neighbours light curves stars = self.download_neighbours(prepare_tic_input.tic, sectors)
            # TODO store folded light curves -with local and global views-(masking previous candidates?)
        except Exception as e:
            logging.exception("Failed long cadence for %s", mission_id)
            failed_targets_long = failed_targets_long + [mission_id]
        if index % 100:
            shutil.rmtree(cache_dir + "/.lightkurve-cache", ignore_errors=True)
            logging.info("Failed short targets %.0f", len(failed_targets_short))
            logging.info("Failed long targets %.0f", len(failed_targets_long))
    np.savetxt(target_dir + 'failed_targets_short.csv', np.array(failed_targets_short), delimiter=',', fmt='%f')
    np.savetxt(target_dir + 'failed_targets_long.csv', np.array(failed_targets_long), delimiter=',', fmt='%f')


def create_star_csvs(lcs_dir, search_radius=50, pixel_size=4, cores=os.cpu_count() - 1):
    create_star_inputs = []
    for lc_file in list(pathlib.Path(lcs_dir).glob('*time_series_long*.csv')):
        print(lc_file)
        create_star_inputs = create_star_inputs + [CreateStarCsvInput(str(lc_file), pixel_size, search_radius)]
    with Pool(processes=cores) as pool:
        pool.map(_create_star_csv, create_star_inputs)


class CreateStarCsvInput:
    def __init__(self, lc_file, pixel_size=4, search_radius=50) -> None:
        super().__init__()
        self.lc_file = lc_file
        self.search_radius = search_radius
        self.pixel_size = pixel_size * u.arcsec


def _create_star_csv(create_star_input):
    tries = 0
    object_id = None
    while tries < 3:
        try:
            lc_filename_match = re.search("KIC ([0-9]+)_time_series_long", create_star_input.lc_file)
            object_id = int(lc_filename_match[1])
            leading_zeros_object_id = '{:09}'.format(object_id)
            star_info = KicStarCatalog().catalog_info(object_id)
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
        print('Something failed when retrieving neighbours info for object id ' + str(object_id))
        traceback.print_exc()
    star_df.to_csv(lcs_dir + '/' + leading_zeros_object_id + '_star.csv')
    print('Processed object id ' + str(object_id))

formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
while len(logger.handlers) > 0:
    logger.handlers.pop()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

root_dir = '/home/mdevora/scratch/ml/q1_q17/'
lcs_dir = root_dir + '/lcs'
#uncompress_data('/mnt/DATA-2/ete6/backebs/BackEBs/')
#uncompress_data('/mnt/DATA-2/ete6/ebs/EBs/')
#uncompress_data('/mnt/DATA-2/ete6/planets/Planets/')
#_create_star_csv(CreateStarCsvInput('/home/mdevora/scratch/ml/q1_q17/lcs/KIC 12106934_time_series_long.csv'))
create_star_csvs(lcs_dir, cores=10)
#create_injection_dataframe(root_dir, lcs_dir)
#create_targets_df(root_dir)
#prepare_labeled_lcs(root_dir + '/classified_tces.csv', target_dir=lcs_dir,
#                    cache_dir="/home/mdevora/scratch/")
#_create_target_csv(CreateTargetInput(lcs_dir + '/tess2019128220341-0000000230382377-0016-s_lc.fits', root_dir, pd.read_csv(root_dir + '/injected_objects.csv')))
