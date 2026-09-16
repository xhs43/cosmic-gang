import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)

def find_fits_files(session_dir, darks_dir):
    science_files = sorted(glob.glob(os.path.join(session_dir, '*.fit*')))
    dark_files = sorted(glob.glob(os.path.join(darks_dir, '*.fit*')))
    return science_files, dark_files

def create_master_dark(dark_files):
    if not dark_files:
        raise ValueError("No dark frames found.")
    darks = []
    for f in dark_files:
        with fits.open(f) as hdul:
            darks.append(hdul[0].data.astype(float))
    darks = np.array(darks)
    master_dark = np.median(darks, axis=0) 
    return master_dark

def compute_airmass(header):
    if 'AIRMASS' in header:
        return header['AIRMASS']
    if 'TELALT' in header:
        alt = header['TELALT']
        if alt > 0:
            return 1.0 / np.sin(np.radians(alt))
    return 1.0

def track_star(daofind_table, expected_x, expected_y, search_radius=45.0):
    """Find the brightest detection within search radius of expected coordinates."""
    if daofind_table is None or len(daofind_table) == 0:
        return None
    
    xs = daofind_table['x_centroid']
    ys = daofind_table['y_centroid']
    peaks = daofind_table['peak']
    dists = np.sqrt((xs - expected_x)**2 + (ys - expected_y)**2)
    
    valid_idx = np.where(dists <= search_radius)[0]
    if len(valid_idx) == 0:
        return None
    
    best_idx = valid_idx[np.argmax(peaks[valid_idx])]
    return float(xs[best_idx]), float(ys[best_idx])


def estimate_fwhm(data, sources, max_stars=10, box_size=7):
    import numpy as np
    fwhms = []
    valid = sources[sources['peak'] < 3500]
    valid.sort('peak')
    valid.reverse()
    for row in valid[:max_stars]:
        x0, y0 = int(round(row['x_centroid'])), int(round(row['y_centroid']))
        if y0-box_size < 0 or y0+box_size >= data.shape[0] or x0-box_size < 0 or x0+box_size >= data.shape[1]: continue
        cutout = data[y0-box_size:y0+box_size+1, x0-box_size:x0+box_size+1]
        cutout = cutout - np.median(cutout)
        peak = cutout.max()
        if peak <= 0: continue
        area = np.sum(cutout >= peak / 2.0)
        fwhms.append(2 * np.sqrt(area / np.pi))
    return np.median(fwhms) if fwhms else 4.0

def process_session(science_dir, darks_dir, target_coords, comp_coords):

    science_files, dark_files = find_fits_files(science_dir, darks_dir)
    print(f"Found {len(science_files)} Science frames and {len(dark_files)} Dark frames.")
    
    if not science_files:
        print("No science frames to process.")
        return
        
    master_dark = create_master_dark(dark_files)
    
    times = []
    fluxes = []
    flux_errs = []
    fwhms = []
    
    obs_location = EarthLocation.from_geodetic(-110.88*u.deg, 31.68*u.deg, 1268*u.m)
    
    current_target = target_coords
    current_comps = list(comp_coords)
    
    # Fix for tracking: Use original target_coords for EVERY frame as expected position
    # Fix for Time: Strip timezone info from DATE-OBS
    # Fix for Negative flux: use absolute value or floor at 0 for error calc, just accept negative flux if sky is bright
    
    measured_frames = 0
    rejected_frames = 0
    first_frame_data = None
    first_frame_apertures = None
    
    for i, sci_file in enumerate(science_files):
        try:
            with fits.open(sci_file) as hdul:
                header = hdul[0].header
                data = hdul[0].data.astype(float) - master_dark
                
                date_obs_raw = header['DATE-OBS']
                # Strip timezone like -0700 or Z
                if '-' in date_obs_raw[10:]: # check after YYYY-MM-DD
                    date_obs = date_obs_raw[:date_obs_raw.rfind('-')]
                elif '+' in date_obs_raw[10:]:
                    date_obs = date_obs_raw[:date_obs_raw.rfind('+')]
                elif date_obs_raw.endswith('Z'):
                    date_obs = date_obs_raw[:-1]
                else:
                    date_obs = date_obs_raw
                    
                exptime = header.get('EXPTIME', 1.0)
                
                # Estimate global FWHM and background
                mean, median, std = sigma_clipped_stats(data, sigma=3.0)
                
                # Detect stars
                daofind = DAOStarFinder(fwhm=4.0, threshold=2. * std)
                sources = daofind(data - median)
                
                if sources is None or len(sources) < 3:
                    print(f"Skipping frame {i}: Not enough stars detected.")
                    rejected_frames += 1
                    continue
                
                fwhm = estimate_fwhm(data, sources)
                ap_radius = max(3.0, 2.5 * fwhm)
                
                # Track stars from original coordinates, using COMP1 for global offset
                new_c1 = track_star(sources, comp_coords[0][0], comp_coords[0][1], search_radius=150.0)
                if new_c1 is None:
                    new_comps = [None, None]
                    new_target = None
                else:
                    dx = new_c1[0] - comp_coords[0][0]
                    dy = new_c1[1] - comp_coords[0][1]
                    new_target = track_star(sources, target_coords[0] + dx, target_coords[1] + dy, search_radius=10.0)
                    new_c2 = track_star(sources, comp_coords[1][0] + dx, comp_coords[1][1] + dy, search_radius=10.0)
                    new_comps = [new_c1, new_c2]
                    
                if new_target is None or None in new_comps:
                    print(f"Skipping frame {i}: Lost tracking (clouds or drift).")
                    rejected_frames += 1
                    # Keep current_target and current_comps unchanged for the next frame
                    continue
                    
                # Update last known positions
                current_target = new_target
                current_comps = new_comps
                
                # Photometry
                positions = [current_target] + current_comps
                apertures = CircularAperture(positions, r=ap_radius)
                annulus_apertures = CircularAnnulus(positions, r_in=ap_radius+5, r_out=ap_radius+15)
                
                phot_table = aperture_photometry(data, apertures)
                bkg_table = aperture_photometry(data, annulus_apertures)
                
                # Background subtraction
                annulus_masks = annulus_apertures.to_mask(method='center')
                bkg_median = []
                for mask in annulus_masks:
                    annulus_data = mask.multiply(data)
                    annulus_data_1d = annulus_data[mask.data > 0]
                    _, bkg_med, _ = sigma_clipped_stats(annulus_data_1d)
                    bkg_median.append(bkg_med)
                
                bkg_median = np.array(bkg_median)
                phot_table['bkg_median'] = bkg_median
                phot_table['aper_bkg'] = bkg_median * apertures.area
                phot_table['aper_sum_bkgsub'] = phot_table['aperture_sum'] - phot_table['aper_bkg']
                
                f_t = phot_table['aper_sum_bkgsub'][0]
                f_c1 = phot_table['aper_sum_bkgsub'][1]
                f_c2 = phot_table['aper_sum_bkgsub'][2]
                
                comp_flux = f_c1 + f_c2
                if not hasattr(process_session, 'max_comp'): process_session.max_comp = 0
                if comp_flux > process_session.max_comp: process_session.max_comp = comp_flux
                if comp_flux <= 0 or f_t <= 0 or comp_flux < 0.5 * process_session.max_comp:
                    print(f"Skipping frame {i}: Unreliable flux.")
                    rejected_frames += 1
                    continue
                
                f_norm = f_t / (f_c1 + f_c2)
                
                # Simple Poisson error estimation
                err_t = np.sqrt(max(f_t, 0) + max(phot_table['aper_bkg'][0], 0))
                err_c1 = np.sqrt(max(f_c1, 0) + max(phot_table['aper_bkg'][1], 0))
                err_c2 = np.sqrt(max(f_c2, 0) + max(phot_table['aper_bkg'][2], 0))
                
                # Error propagation for F_t / (F_c1 + F_c2)
                err_c12 = np.sqrt(err_c1**2 + err_c2**2)
                f_err = abs(f_norm) * np.sqrt((err_t/max(abs(f_t), 1))**2 + (err_c12/max(abs(f_c1+f_c2), 1))**2)
                
                # BJD_TDB
                t = Time(date_obs, format='isot', scale='utc', location=obs_location)
                ra = header.get('RA', 286.04)
                dec = header.get('DEC', 36.63)
                target_sky = SkyCoord(ra, dec, unit=(u.deg, u.deg))
                ltt_bary = t.light_travel_time(target_sky)
                bjd_tdb = (t.tdb + ltt_bary).jd
                
                times.append(bjd_tdb)
                fluxes.append(f_norm)
                flux_errs.append(f_err)
                fwhms.append(fwhm)
                
                measured_frames += 1
                
                if first_frame_data is None:
                    first_frame_data = data
                    first_frame_apertures = positions
                    
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            rejected_frames += 1
            
    print(f"\nProcessing complete:")
    print(f"- Successfully measured: {measured_frames}")
    print(f"- Rejected: {rejected_frames}")
    if fwhms:
        print(f"- Median FWHM: {np.median(fwhms):.2f} px")
        print(f"- Aperture radius: {2.5 * np.median(fwhms):.2f} px")
    
    # Baseline normalization
    fluxes = np.array(fluxes)
    flux_errs = np.array(flux_errs)
    baseline = np.median(fluxes)
    if baseline > 0:
        fluxes = fluxes / baseline
        flux_errs = flux_errs / baseline

    # Save CSV
    df = pd.DataFrame({
        "Time_BJD": times,
        "Normalized_Flux": fluxes,
        "Flux_Error": flux_errs
    })
    
    out_dir = "processed_lightcurves"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "TRES-1_calibrated.csv")
    df.to_csv(out_csv, index=False)
    print(f"- Output CSV: {out_csv}")
    
    # Generate diagnostic plots
    if measured_frames > 0:
        # Plot 1: The calibrated frame with apertures
        fig, ax = plt.subplots(figsize=(8, 6))
        m, s = np.nanmean(first_frame_data), np.nanstd(first_frame_data)
        im = ax.imshow(first_frame_data, interpolation='nearest', cmap='viridis',
                       vmin=m-s, vmax=m+5*s, origin='lower')
        
        ap_r = 2.5 * 4.0
        colors = ['red', 'cyan', 'lime']
        labels = ['Target', 'Comp1', 'Comp2']
        for pos, color, label in zip(first_frame_apertures, colors, labels):
            c = Circle(pos, ap_r, color=color, fill=False, lw=2)
            ax.add_patch(c)
            ax.text(pos[0]+15, pos[1]+15, label, color=color, fontsize=12)
            
        plt.title('Aperture Overlay - First Frame')
        plt.savefig(os.path.join(out_dir, "aperture_overlay.png"))
        plt.close()
        
        # Plot 2: The Light Curve
        plt.figure(figsize=(10, 5))
        plt.errorbar(times, fluxes, yerr=flux_errs, fmt='o', color='black', markersize=3, alpha=0.7)
        plt.xlabel('BJD_TDB')
        plt.ylabel('Normalized Flux')
        plt.title('TrES-1 Differential Light Curve')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "lightcurve.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--science_dir", type=str, required=True)
    parser.add_argument("--darks_dir", type=str, required=True)
    parser.add_argument("--target_x", type=float, required=True)
    parser.add_argument("--target_y", type=float, required=True)
    parser.add_argument("--comp1_x", type=float, required=True)
    parser.add_argument("--comp1_y", type=float, required=True)
    parser.add_argument("--comp2_x", type=float, required=True)
    parser.add_argument("--comp2_y", type=float, required=True)
    args = parser.parse_args()
    
    target_coords = (args.target_x, args.target_y)
    comp_coords = [(args.comp1_x, args.comp1_y), (args.comp2_x, args.comp2_y)]
        
    process_session(args.science_dir, args.darks_dir, target_coords, comp_coords)
