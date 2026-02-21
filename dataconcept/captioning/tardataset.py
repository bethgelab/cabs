import tarfile
from io import BytesIO
from PIL import Image, ImageFile
import json

from torch.utils.data import Dataset, get_worker_info

try:  # make torchvision optional
    from torchvision.transforms.functional import to_tensor
except:
    to_tensor = None

ImageFile.LOAD_TRUNCATED_IMAGES = True


class UnexpectedEOFTarFile(tarfile.TarFile):
    def _load(self):
        """Read through the entire archive file and look for readable
           members.
        """
        try:
            while True:
                tarinfo = self.next()
                if tarinfo is None:
                    break
        except tarfile.ReadError as e:
            assert e.args[0] == "unexpected end of data"
        self._loaded = True


class TarDatasetBase(Dataset):
    """Dataset that supports Tar archives (uncompressed).

    Args:
      archive (string or TarDataset): Path to the Tar file containing the dataset.
        Alternatively, pass in a TarDataset object to reuse its cached information;
        this is useful for loading different subsets within the same archive.
      extensions (tuple): Extensions (strings starting with a dot), only files
        with these extensions will be iterated. Default: png/jpg/jpeg.
      is_valid_file (callable): Optional function that takes file information as
        input (tarfile.TarInfo) and outputs True for files that need to be
        iterated; overrides extensions argument.
        Example: lambda m: m.isfile() and m.name.endswith('.png')
      transform (callable): Function applied to each image by __getitem__ (see
        torchvision.transforms). Default: ToTensor (convert PIL image to tensor).
      ignore_unexpected_eof (bool): ignore Unexpected EOF when iterating the tar file
        allows working with Datasets cut to a smaller size with dd

    Attributes:
      members_by_name (dict): Members (files and folders) found in the Tar archive,
        with their names as keys and their tarfile.TarInfo structures as values.
      samples (list): Items to iterate (can be ignored by overriding __getitem__
        and __len__).

    Author: Joao F. Henriques
    """

    def __init__(self, archive, transform=to_tensor, tagset=None, extensions=('.png', '.jpg', '.jpeg'),
                 is_valid_file=None, image_size=100, ignore_unexpected_eof=False, assume_jsons=True):
        if not isinstance(archive, TarDatasetBase):
            # open tar file. in a multiprocessing setting (e.g. DataLoader workers), we
            # have to open one file handle per worker (stored as the tar_obj dict), since
            # when the multiprocessing method is 'fork', the workers share this TarDataset.
            # we want one file handle per worker because TarFile is not thread-safe.
            worker = get_worker_info()
            worker = worker.id if worker else None
            self.tar_obj = {
                worker: tarfile.open(archive) if ignore_unexpected_eof is False else UnexpectedEOFTarFile.open(archive)}
            self.archive = archive
            self.assume_jsons = assume_jsons
            self.image_size = image_size

            # store headers of all files and folders by name
            # members = sorted(self.tar_obj[worker].getmembers(), key=lambda m: m.name)
            try:
                members = sorted(self.tar_obj[worker].getmembers(), key=lambda m: m.name)
            except tarfile.ReadError as e:
                print(f"Corrupted tar file: {e}. Skipping this file.")
                members = []
            self.members_by_name = {m.name: m for m in members}
        else:
            # passed a TarDataset into the constructor, reuse the same tar contents.
            # no need to copy explicitly since this dict will not be modified again.
            self.members_by_name = archive.members_by_name
            self.archive = archive.archive  # the original path to the Tar file
            self.tar_obj = {}  # will get filled by get_file on first access

        # also store references to the iterated samples (a subset of the above)
        self.filter_samples(is_valid_file, extensions)
        self.tagset = tagset
        self.transform = transform

    def filter_samples(self, is_valid_file=None, extensions=('.png', '.jpg', '.jpeg')):
        """Filter the Tar archive's files/folders to obtain the list of samples.

        Args:
          extensions (tuple): Extensions (strings starting with a dot), only files
            with these extensions will be iterated. Default: png/jpg/jpeg.
          is_valid_file (callable): Optional function that takes file information as
            input (tarfile.TarInfo) and outputs True for files that need to be
            iterated; overrides extensions argument.
            Example: lambda m: m.isfile() and m.name.endswith('.png')
        """
        # by default, filter files by extension
        # Assume that all jpgs have associated json file
        if is_valid_file is None:
            def is_valid_file(m):
                return (m.isfile() and m.name.lower().endswith(extensions))

        # filter the files to create the samples list
        self.samples = [m.name for m in self.members_by_name.values() if is_valid_file(m)]

    def __getitem__(self, index):
        """
        For the given index, load the image from the image tar and the corresponding metadata
        from the metadata tar. This logic mirrors process_images_and_metadata.
        """
        # Get the jpg file name from the image tar samples
        image_name = self.samples[index]

        image = self.get_image(image_name)

        if image is None:
            print(f"Skipping image {image_name} due to loading error")
            return None

        json_name = image_name.rsplit('.', 1)[0] + '.json'

        try:
            json_data = json.loads(self.get_text_file(json_name))
            assert json_data is not None, 'json {} is None'.format(json_name)
        except Exception as e:
            print(f"Error parsing JSON {json_name}: {e}")
            json_data = {}

        # Extract desired fields from the metadata
        caption = json_data.get('caption', None)

        if self.transform:
            image = self.transform(image)
        return (image_name, image, caption)

    def __len__(self):
        """Return the length of the dataset (length of self.samples)

        Returns:
          int: Number of samples.
        """
        return len(self.samples)

    def get_image(self, name):
        """Read an image from the Tar archive, returned as a PIL image or PyTorch tensor.

        Args:
          name (str): File name to retrieve.
          pil (bool): If true, a PIL image is returned (default is a PyTorch tensor).

        Returns:
          Image or Tensor: The image, possibly in PIL format.
        """
        # image = Image.open(BytesIO(self.get_file(name).read()))
        # return image
        try:
            image = Image.open(BytesIO(self.get_file(name).read()))
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            return image
        except Image.DecompressionBombError:
            print(f"Skipping image {name} due to DecompressionBombError")
            return None

    def get_text_file(self, name, encoding='utf-8'):
        """Read a text file from the Tar archive, returned as a string.

        Args:
          name (str): File name to retrieve.
          encoding (str): Encoding of file, default is utf-8.

        Returns:
          str: Content of text file.
        """
        return self.get_file(name).read().decode(encoding)

    def get_file(self, name):
        """Read an arbitrary file from the Tar archive.

        Args:
          name (str): File name to retrieve.

        Returns:
          io.BufferedReader: Object used to read the file's content.
        """
        # ensure a unique file handle per worker, in multiprocessing settings
        worker = get_worker_info()
        worker = worker.id if worker else None

        if worker not in self.tar_obj:
            self.tar_obj[worker] = tarfile.open(self.archive)

        return self.tar_obj[worker].extractfile(self.members_by_name[name])

    def __del__(self):
        """Close the TarFile file handles on exit."""
        for o in self.tar_obj.values():
            o.close()

    def __getstate__(self):
        """Serialize without the TarFile references, for multiprocessing compatibility."""
        state = dict(self.__dict__)
        state['tar_obj'] = {}
        return state