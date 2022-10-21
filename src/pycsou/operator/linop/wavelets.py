import numpy as np
import pywt

import pycsou.abc as pyca
import pycsou.operator as pycop
import pycsou.operator.linop as pyclop
import pycsou.util.ptype as pyct


class WaveletDec(pyca.LinOp):
    def __init__(self, input_shape, wavelet_name: str, level=None, mode="zero"):
        self.image_shape = input_shape
        self._wavelet = wavelet_name
        self._level = level
        self._mode = mode
        init_coeffs = pywt.wavedec2(np.zeros(self.image_shape), self._wavelet, mode=self._mode, level=self._level)
        arr, slices = pywt.coeffs_to_array(init_coeffs, axes=(-2, -1))
        self._coeffs_slices = slices
        self.coeffs_shape = arr.shape
        super().__init__(shape=(arr.shape[0] * arr.shape[1], input_shape[0] * input_shape[1]))
        if self._mode == "zero":
            self._lipschitz = 1.0

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        raster_image = arr.reshape(*arr.shape[:-1], *self.image_shape)
        coeffs = pywt.wavedec2(
            raster_image,
            self._wavelet,
            mode=self._mode,
            level=self._level,
        )
        return pywt.coeffs_to_array(coeffs, axes=(-2, -1))[0].reshape(*arr.shape[:-1], -1)

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        raster_arr = arr.reshape(*arr.shape[:-1], *self.coeffs_shape)
        if arr.ndim == 1:
            slices = self._coeffs_slices
        else:
            _, slices = pywt.coeffs_to_array(
                pywt.wavedec2(np.zeros((*arr.shape[:-1], *self.image_shape)), self._wavelet, level=self._level),
                axes=(-2, -1),
            )
        coeffs = pywt.array_to_coeffs(raster_arr, slices, output_format="wavedec2")
        res = pywt.waverec2(
            coeffs,
            self._wavelet,
            mode=self._mode,
        )
        return res.reshape(*arr.shape[:-1], -1)


def stackedWaveletDec(input_shape, wl_list, level_list, mode_list="zero", include_dirac: bool = True) -> pyca.LinOp:
    length = len(wl_list) + include_dirac
    if not isinstance(level_list, list):
        level_list = [
            level_list,
        ] * length
    if not isinstance(mode_list, list):
        mode_list = [
            mode_list,
        ] * length
    op_list = [
        WaveletDec(input_shape, wl, level=level, mode=mode) for wl, level, mode in zip(wl_list, level_list, mode_list)
    ]
    if include_dirac:
        op_list.append(pyclop.IdentityOp(dim=input_shape[0] * input_shape[1]))

    return (1.0 / np.sqrt(length)) * pycop.stack(op_list, axis=0)
