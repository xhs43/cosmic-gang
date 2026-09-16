import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
import scipy.signal as signal

class TransitDetector:
    def __init__(self, min_period=0.5, max_period=15.0, signal_to_noise_threshold=5.0):
        """
        تهيئة أداة الاكتشاف واختيار حد العتبة لنسبة الإشارة إلى الضوضاء
        """
        self.min_period = min_period
        self.max_period = max_period
        self.snr_threshold = signal_to_noise_threshold

    def preprocess_lightcurve(self, time, flux):
        """
        تنظيف بيانات المنحنى الضوئي وإزالة الاتجاهات العامة (Detrending)
        """
        # إزالة النقاط الشاذة (Outliers)
        median_flux = np.median(flux)
        std_flux = np.std(flux)
        valid_mask = np.abs(flux - median_flux) < (3 * std_flux)
        
        clean_time = time[valid_mask]
        clean_flux = flux[valid_mask]
        
        # تطبيق فلتر Savitzky-Golay لتنعيم المنحنى
        window_length = min(len(clean_flux) // 10 | 1, 101)
        if window_length > 3:
            trend = signal.savgol_filter(clean_flux, window_length=window_length, polyorder=2)
            normalized_flux = clean_flux / trend
        else:
            normalized_flux = clean_flux / median_flux
            
        return clean_time, normalized_flux

    def detect_transit(self, time, flux, flux_err=None):
        """
        تطبيق خوارزمية BLS المباشرة لحساب التدفق وإمكانية الاكتشاف
        """
        c_time, c_flux = self.preprocess_lightcurve(time, flux)
        model = BoxLeastSquares(c_time, c_flux, dy=flux_err)
        
        periods = np.linspace(self.min_period, self.max_period, 10000)
        durations = np.linspace(0.05, 0.2, 10)
        
        results = model.power(periods, durations)
        best_idx = np.argmax(results.power)
        
        max_power = results.power[best_idx]
        mean_power = np.mean(results.power)
        std_power = np.std(results.power)
        
        # حساب نسبة الإشارة إلى الضوضاء (SNR)
        snr = (max_power - mean_power) / std_power if std_power > 0 else 0
        
        return {
            "detected": bool(snr >= self.snr_threshold),
            "snr": float(snr),
            "period": float(results.period[best_idx]),
            "transit_time": float(results.transit_time[best_idx]),
            "depth": float(results.depth[best_idx]),
            "power": float(max_power)
        }

if __name__ == "__main__":
    # تشغيل تجريبي برلمي لسلسلة زمنية افتراضية
    t = np.linspace(0, 10, 1000)
    f = np.ones_like(t)
    f[(t % 3.0 > 1.2) & (t % 3.0 < 1.35)] -= 0.015
    f += np.random.normal(0, 0.002, size=len(t))
    
    detector = TransitDetector()
    print("نتيجة الاكتشاف السريعة:", detector.detect_transit(t, f))