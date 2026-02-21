import json
from tardataset import TarDatasetBase


class TarDataset(TarDatasetBase):
    """extends TarDatasetBase to also yield detected object classes."""

    def __getitem__(self, index):
        image_name = self.samples[index]
        image = self.get_image(image_name)
        if image is None:
            return None

        json_name = image_name.rsplit('.', 1)[0] + '.json'
        try:
            json_data = json.loads(self.get_text_file(json_name))
        except Exception as e:
            print(f"error parsing {json_name}: {e}")
            json_data = {}

        caption = json_data.get('caption', None)
        classes = json_data.get('classes', [])

        if self.transform:
            image = self.transform(image)
        return image_name, image, caption, classes


def collate_fn(batch):
    """filter out failed samples and group by field."""
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    return tuple(zip(*valid))
