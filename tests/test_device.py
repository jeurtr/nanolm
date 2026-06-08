"""验证 nanolm.device 设备检测工具"""

from unittest.mock import patch

import torch

from nanolm.device import (
    empty_cache,
    get_optimal_device,
    resolve_device_type,
)


def test_get_optimal_device_cpu():
    """没有 CUDA 和 MPS 时返回 cpu"""
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps', create=True) as mock_mps:
        mock_mps.is_available.return_value = False
        assert get_optimal_device() == 'cpu'


def test_get_optimal_device_falls_back_to_cpu():
    """检测不到任何加速设备时返回 cpu"""
    result = get_optimal_device()
    assert result in ('cpu', 'cuda', 'mps')


def test_resolve_device_type_str():
    assert resolve_device_type('cuda') == 'cuda'
    assert resolve_device_type('cpu') == 'cpu'


def test_resolve_device_type_mps_returns_cpu():
    """MPS 不支持 autocast，应返回 cpu"""
    assert resolve_device_type('mps') == 'cpu'


def test_resolve_device_type_torch_device():
    assert resolve_device_type(torch.device('cpu')) == 'cpu'
    assert resolve_device_type(torch.device('cuda:0')) == 'cuda'
    assert resolve_device_type(torch.device('mps')) == 'cpu'


def test_resolve_device_type_int():
    assert resolve_device_type(0) == '0'


def test_empty_cache_noop():
    """empty_cache 在无 CUDA 时不报错"""
    with patch('torch.cuda.is_available', return_value=False):
        empty_cache()  # 不应抛出异常


def test_empty_cache_calls_cuda():
    """CUDA 可用时调用 torch.cuda.empty_cache"""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.empty_cache') as mock_empty:
        empty_cache()
        mock_empty.assert_called_once()
