from torchmetrics.image.fid import FrechetInceptionDistance
import torch

class MyFID():
    def __init__(self, device, features=2048, normalize=True):

        self.fid = FrechetInceptionDistance(feature=features, normalize=normalize).to(device)
    
    def normalize_batch(self, batch):
        # FID implementation wnat images to be in range 0-255 or 0-1
        normalized_images = []
        for image in batch:
            image = image.squeeze(0)
        
            min_val = image.amin()
            max_val = image.amax()

            normalized_image = (image - min_val) / (max_val - min_val)
            normalized_images.append(normalized_image.unsqueeze(0))
        
        return torch.stack(normalized_images)

    
    def gray_to_rgb(self, real_batch, fake_batch):
        # insecption network works with RGB images
        
        rgb_real = real_batch.repeat(1, 3, 1, 1)
        rgb_fake = fake_batch.repeat(1, 3, 1, 1)

        return rgb_real, rgb_fake
    

    def get_distance(self, real_batch, fake_batch):
        real_images = self.normalize_batch(real_batch)
        fake_images = self.normalize_batch(fake_batch)

        real_images, fake_images = self.gray_to_rgb(real_images, fake_images)

        self.fid.update(fake_images, real=False)
        self.fid.update(real_images, real=True)
        distance = self.fid.compute()
        self.fid.reset()

        return distance.item()