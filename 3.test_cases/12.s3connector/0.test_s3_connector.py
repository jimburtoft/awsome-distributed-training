import s3torchconnector

from PIL import Image
import webdataset
import itertools
import io

def shard_to_dict(object):
    return {"url": object.key, "stream": object}



IMAGES_URI = "s3://laion-art/laion-art-data/tarfiles_reorganized/task0000/"

REGION = "us-west-2"

dataset = s3torchconnector.S3IterableDataset.from_prefix(IMAGES_URI, region=REGION, transform=shard_to_dict)

dataset = webdataset.tariterators.tar_file_expander(dataset)

for sample in itertools.islice(dataset, 0, 100, 20):
    print(sample["fname"])
    img = Image.open(io.BytesIO(sample["data"])).reduce(8)
    img.save(sample["fname"])

loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
for keys, images in itertools.islice(loader, 3):
    print(keys)
    print(images.shape)


## https://medium.com/red-buffer/why-did-i-choose-webdataset-for-training-on-50tb-of-data-98a563a916bf

## https://github.com/pytorch/pytorch/issues/38419