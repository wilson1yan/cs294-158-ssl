import requests
from collections import OrderedDict, Counter
import torch


def to_one_hot(labels, d, device):
    one_hot = torch.FloatTensor(labels.shape[0], d).to(device)
    one_hot.zero_()
    one_hot.scatter_(1, labels.unsqueeze(1), 1)

    return one_hot


def append_location(x, device):
    idxs = torch.arange(28).float() / 27  # Scale to [0, 1]
    locs = torch.stack(torch.meshgrid(idxs, idxs), dim=-1)
    locs = locs.permute(2, 0, 1).contiguous().unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
    locs = locs.to(device)

    x = torch.cat((x, locs), dim=1)
    return x


# Sample from multinomial distribution using gumbel-max
def sample_multinomial(logits, dim=1):
    logits = torch.log_softmax(logits, dim=dim)
    # Clamp for stability
    u = torch.clamp(torch.rand_like(logits), 1e-5, 1 - 1e-5)
    gumbel = -torch.log(-torch.log(u))
    sample = torch.max(logits + gumbel, dim=1)[1]
    return sample.float()


def quantize(img, n_bits):
    n_colors = 2 ** n_bits
    # Quantize to integers from 0, ..., n_colors - 1
    img = torch.clamp(torch.floor((img * n_colors)), max=n_colors - 1)
    img /= n_colors - 1 # Scale to [0, 1]
    return img


def to_grayscale(img):
    if len(img.shape) == 3:
        return 0.3 * img[[0]] + 0.59 * img[[1]] + 0.11 * img[[2]]
    elif len(img.shape) == 4:
        return 0.3 * img[:, [0]] + 0.59 * img[:, [1]] + 0.11 * img[:, [2]]
    else:
        raise Exception('Invalid img shape', img.shape)


def download_file(url, filename):
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)


# Code to download from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

