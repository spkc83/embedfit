import io
import struct

import numpy as np
import pytest
import torch
from PIL import Image

from embedfit.data import image as image_module
from embedfit.data import to_rgb_tensor
from embedfit.data.image import is_readable
from embedfit.exceptions import ImageInputError


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGB", (48, 20)),
        Image.new("L", (48, 20)),
        Image.new("RGBA", (48, 20)),
        np.zeros((20, 48), dtype=np.uint8),
        np.zeros((20, 48, 1), dtype=np.uint8),
        np.zeros((20, 48, 3), dtype=np.float32),
        torch.zeros(20, 48),
        torch.zeros(1, 20, 48, dtype=torch.uint8),
        torch.zeros(3, 20, 48),
    ],
    ids=[
        "pil-rgb",
        "pil-gray",
        "pil-rgba",
        "np-hw",
        "np-hw1",
        "np-hw3-float",
        "t-hw",
        "t-1hw-uint8",
        "t-3hw",
    ],
)
def test_preprocess_accepts_supported_inputs(image):
    pixels = to_rgb_tensor(image)
    assert pixels.shape == (3, 20, 48)  # Native resolution is kept.
    assert pixels.dtype == torch.float32


def test_preprocess_preserves_axis_and_channel_layout():
    # Every channel and both spatial axes are distinguishable, so a transposed or
    # channel-reversed conversion cannot pass.
    rows = torch.arange(32, dtype=torch.float32).view(32, 1).expand(32, 32)
    cols = torch.arange(32, dtype=torch.float32).view(1, 32).expand(32, 32)
    expected = torch.stack([rows / 31, cols / 31, torch.full((32, 32), 0.25)])
    hwc_uint8 = (expected.permute(1, 2, 0) * 255).round().to(torch.uint8)
    expected_uint8 = hwc_uint8.permute(2, 0, 1).float() / 255

    assert torch.equal(to_rgb_tensor(expected), expected)
    assert torch.equal(to_rgb_tensor(expected.permute(1, 2, 0).numpy()), expected)
    assert torch.equal(to_rgb_tensor(hwc_uint8.numpy()), expected_uint8)
    pil = Image.fromarray(hwc_uint8.numpy())
    assert torch.equal(to_rgb_tensor(pil), expected_uint8)


def test_preprocess_never_aliases_or_tracks_the_input():
    source = torch.full((3, 32, 32), 0.5, requires_grad=True)
    pixels = to_rgb_tensor(source)
    assert pixels.data_ptr() != source.data_ptr()
    assert not pixels.requires_grad

    read_only = np.full((32, 32, 3), 0.5, dtype=np.float32)
    read_only.setflags(write=False)
    to_rgb_tensor(read_only).zero_()
    assert read_only.min() == 0.5


def test_preprocess_scales_uint8_and_replicates_grayscale():
    pixels = to_rgb_tensor(np.full((32, 32), 255, dtype=np.uint8))
    assert torch.allclose(pixels, torch.ones(3, 32, 32))


@pytest.mark.parametrize(
    "bad, match",
    [
        (12345, "Unsupported image type"),
        (np.zeros((2, 8, 8, 3), dtype=np.uint8), "list of images"),
        (np.zeros((4, 4, 3, 1), dtype=np.uint8), "dimensions"),
        (np.zeros((8, 8, 2), dtype=np.uint8), "1 or 3 channels"),
        (torch.zeros(4, 8, 8), "1 or 3 channels"),
        (np.zeros((8, 8), dtype=np.int64), "dtype"),
        (np.zeros((0, 8), dtype=np.uint8), "empty"),
        (np.full((8, 8, 3), 200.0, dtype=np.float32), r"lie in \[0, 1\]"),
        (np.full((8, 8), -0.1), r"lie in \[0, 1\]"),
        (torch.full((3, 8, 8), float("nan")), "finite"),
    ],
)
def test_preprocess_rejects_malformed_inputs(bad, match):
    with pytest.raises(ImageInputError, match=match):
        to_rgb_tensor(bad)


def test_error_messages_do_not_leak_pixel_values():
    secret = np.full((8, 8, 2), 173, dtype=np.uint8)
    with pytest.raises(ImageInputError) as excinfo:
        to_rgb_tensor(secret)
    assert "173" not in str(excinfo.value)


def test_multiframe_image_rejected(tmp_path):
    frames = [Image.new("L", (16, 16), color=value) for value in (0, 255)]
    path = tmp_path / "frames.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:])
    with Image.open(path) as image, pytest.raises(ImageInputError, match="Multi-frame"):
        to_rgb_tensor(image)


def test_file_path_matches_in_memory_image(tmp_path):
    image = Image.fromarray(np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3) % 251)
    image.save(tmp_path / "image.png")
    expected = to_rgb_tensor(image)
    assert torch.equal(to_rgb_tensor(tmp_path / "image.png"), expected)
    assert torch.equal(to_rgb_tensor(str(tmp_path / "image.png")), expected)


def test_missing_and_corrupt_files_raise_without_leaking_the_path(tmp_path):
    secret_name = "acct-99887766.png"
    with pytest.raises(ImageInputError, match="does not exist") as missing:
        to_rgb_tensor(tmp_path / secret_name)
    (tmp_path / secret_name).write_bytes(b"this is not an image")
    with pytest.raises(ImageInputError, match="could not be decoded") as corrupt:
        to_rgb_tensor(tmp_path / secret_name)
    for excinfo in (missing, corrupt):
        assert "99887766" not in str(excinfo.value)


def test_16_bit_grayscale_keeps_its_full_range(tmp_path):
    levels = np.linspace(0, 65535, 256).astype(np.uint16).reshape(16, 16)
    Image.fromarray(levels).save(tmp_path / "scan16.png")
    pixels = to_rgb_tensor(tmp_path / "scan16.png")
    assert pixels.shape == (3, 16, 16)
    assert torch.allclose(pixels[0], torch.from_numpy(levels / 65535.0).float(), atol=1e-6)
    assert pixels.unique().numel() == 256


def test_float_mode_pil_image_is_validated_not_clipped():
    ramp = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    assert torch.allclose(to_rgb_tensor(Image.fromarray(ramp))[0], torch.from_numpy(ramp))
    with pytest.raises(ImageInputError, match=r"lie in \[0, 1\]"):
        to_rgb_tensor(Image.fromarray(ramp * 255))


def test_32_bit_integer_mode_is_rejected():
    with pytest.raises(ImageInputError, match="mode 'I'"):
        to_rgb_tensor(Image.new("I", (8, 8)))


def test_exif_orientation_is_applied(tmp_path):
    image = Image.new("RGB", (40, 20))
    exif = Image.Exif()
    exif[0x0112] = 6  # Rotate 90 degrees clockwise to display.
    image.save(tmp_path / "rotated.jpg", exif=exif)
    assert to_rgb_tensor(tmp_path / "rotated.jpg").shape == (3, 40, 20)


@pytest.mark.parametrize("empty", [np.zeros((8, 8, 0), dtype=np.float32), torch.zeros(0, 8, 8)])
def test_zero_channel_input_is_an_image_error(empty):
    with pytest.raises(ImageInputError, match="1 or 3 channels"):
        to_rgb_tensor(empty)


def _encoded(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


def test_encoded_bytes_match_the_same_file(tmp_path):
    image = Image.fromarray(np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3) % 251)
    image.save(tmp_path / "image.png")
    assert torch.equal(to_rgb_tensor(_encoded(image)), to_rgb_tensor(tmp_path / "image.png"))


def test_corrupt_bytes_raise_without_content_or_cause():
    with pytest.raises(ImageInputError, match="bytes could not be decoded") as excinfo:
        to_rgb_tensor(b"acct-99887766 is not an image")
    assert "99887766" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__


def test_unreadable_location_is_not_reported_as_corrupt(tmp_path):
    with pytest.raises(ImageInputError, match="could not be opened"):
        to_rgb_tensor(tmp_path)  # A directory.


def test_decompression_bomb_is_an_image_error(tmp_path, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    Image.new("RGB", (16, 16)).save(tmp_path / "large.png")
    for source in (tmp_path / "large.png", _encoded(Image.new("RGB", (16, 16)))):
        with pytest.raises(ImageInputError, match="pixel limit") as excinfo:
            to_rgb_tensor(source)
        assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__


class TestIsReadable:
    def test_readable_sources(self, tmp_path):
        image = Image.new("RGB", (8, 8))
        image.save(tmp_path / "ok.png")
        frames = [Image.new("L", (8, 8), color) for color in (0, 255)]
        frames[0].save(tmp_path / "pages.tiff", save_all=True, append_images=frames[1:])
        # A multi-frame file decodes; rejecting it is a policy of to_rgb_tensor, not corruption.
        for source in (tmp_path / "ok.png", str(tmp_path / "ok.png"), _encoded(image)):
            assert is_readable(source)
        assert is_readable(tmp_path / "pages.tiff")

    @pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
    def test_missing_garbage_and_truncated_sources(self, tmp_path, fmt):
        data = _encoded(Image.fromarray(np.full((32, 32, 3), 7, np.uint8)), fmt)
        (tmp_path / "garbage.png").write_bytes(b"not an image")
        (tmp_path / "truncated").write_bytes(data[: len(data) // 2])
        for source in (
            tmp_path / "missing.png",
            tmp_path / "garbage.png",
            tmp_path / "truncated",
            b"not an image",
            data[: len(data) // 2],
        ):
            assert not is_readable(source)

    def test_decompression_bomb_is_unreadable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
        Image.new("RGB", (16, 16)).save(tmp_path / "large.png")
        assert not is_readable(tmp_path / "large.png")

    def test_directories_and_permission_errors_still_raise(self, tmp_path, monkeypatch):
        with pytest.raises(ImageInputError, match="could not be opened"):
            is_readable(tmp_path)

        def denied(*args, **kwargs):
            raise PermissionError(13, "Permission denied", str(tmp_path / "acct-99887766.png"))

        monkeypatch.setattr(Image, "open", denied)
        with pytest.raises(ImageInputError, match="could not be opened") as excinfo:
            is_readable(tmp_path / "acct-99887766.png")
        assert "99887766" not in str(excinfo.value) and excinfo.value.__cause__ is None


class _FailingImage:
    """Stands in for a PIL image whose plugin fails while decoding malformed data."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def load(self):
        raise self.error

    def __getattr__(self, name):
        raise self.error


@pytest.mark.parametrize(
    "error",
    [
        ValueError("tile cannot extend outside image"),
        SyntaxError("not a TIFF file"),
        EOFError(),
        struct.error("unpack requires a buffer of 4 bytes"),
        IndexError("list index out of range"),
    ],
)
def test_plugin_decode_errors_are_corruption(tmp_path, monkeypatch, error):
    (tmp_path / "page.tiff").write_bytes(b"II*\x00")
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: _FailingImage(error))
    for source in (tmp_path / "page.tiff", b"II*\x00"):
        assert not is_readable(source)
        with pytest.raises(ImageInputError, match="could not be decoded") as excinfo:
            to_rgb_tensor(source)
        assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__


def test_data_that_loads_but_fails_to_convert_is_corruption(tmp_path, monkeypatch):
    # Some malformed TIFFs load, then fail in the pixel conversion with a TypeError.
    Image.new("RGB", (8, 8)).save(tmp_path / "page.png")

    def fail(image):
        raise TypeError("Missing dimensions")

    monkeypatch.setattr(image_module, "_pil_to_array", fail)
    assert not is_readable(tmp_path / "page.png")
    with pytest.raises(ImageInputError, match="could not be decoded") as excinfo:
        to_rgb_tensor(tmp_path / "page.png")
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__


def test_policy_errors_are_not_reported_as_corruption(tmp_path):
    frames = [Image.new("L", (8, 8), color) for color in (0, 255)]
    frames[0].save(tmp_path / "pages.tiff", save_all=True, append_images=frames[1:])
    with pytest.raises(ImageInputError, match="Multi-frame"):
        to_rgb_tensor(tmp_path / "pages.tiff")


def test_between_one_and_two_times_the_pixel_limit_is_accepted_with_pillows_warning(
    tmp_path, monkeypatch
):
    # Pillow warns above MAX_IMAGE_PIXELS and raises only above twice that.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 200)
    Image.new("RGB", (16, 16)).save(tmp_path / "large.png")  # 256 pixels.
    with pytest.warns(Image.DecompressionBombWarning):
        assert is_readable(tmp_path / "large.png")
    with pytest.warns(Image.DecompressionBombWarning):
        assert to_rgb_tensor(tmp_path / "large.png").shape == (3, 16, 16)
