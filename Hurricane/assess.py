import numpy as np

def PSNR(data, dec_data):
        data_range = np.max(data) - np.min(data)
        diff = data - dec_data
        rmse = np.sqrt(np.mean(diff**2))
        psnr = 20 * np.log10(data_range / rmse)
        return psnr, rmse

