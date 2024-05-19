import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
from torchvision.utils import save_image, make_grid
import random
import os
import pandas as pd
import sys
FOLDER_PATH = "\\".join(os.getcwd().split('\\')[:-1])
sys.path.append(FOLDER_PATH)
from FID.FID import MyFID

from Generator import Generator
from Discriminator import Discriminator
from Dataset import LungsDataset
from Utils import seed_everything, initialize_weights


def saveModel(epoch, generator_state, gen_optimizer_state, gen_save_path, critic_state, cri_ptimizer_state, cri_save_path):

    checkpoint_generator = {
        "epoch": epoch + 1,
        "model_state": generator_state,
        "optim_state": gen_optimizer_state
        }
    torch.save(checkpoint_generator, gen_save_path)

    checkpoint_critic = {
        "epoch": epoch + 1,
        "model_state": critic_state,
        "optim_state": cri_ptimizer_state
        }
    torch.save(checkpoint_critic, cri_save_path)    

def saveImages(fixed_noise, generator, image_path, epoch):
    print("Saving images.")
    with torch.no_grad():
        fake = generator(fixed_noise)
        img_grid_fake = torchvision.utils.make_grid(fake[:], normalize=True)

        file_path_image = image_path + f"Fake_image-{epoch}.png"
        torchvision.utils.save_image(img_grid_fake, file_path_image)

def fidEval(fid, generator, loader, z_dim):
    distances = []
    for _, data in enumerate(loader, 0):
        real_images = data
        real_images = real_images.to(device)
        batch_size = real_images.shape[0]

        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_images = generator(noise)

        distance = fid.get_distance(real_images, fake_images)

        distances.append(distance)
    
    final_distance = sum(distances) / len(distances)

    return final_distance

def saveToCsv(descriptions):
    csv_path = './experiments/experimentInfo.csv'
    if not os.path.isfile(csv_path):
        df = pd.DataFrame([descriptions])
    else:
        df = pd.read_csv(csv_path, index_col=0)
        
        df_e = pd.DataFrame([descriptions])
        df = pd.concat([df, df_e])
    
    df.to_csv(csv_path, index_label=False)

def trainModel(device, learning_rate, fixed_noise, epochs, z_dim, fid):

    # create folder where specific experiment will be saved
    base_path = "./experiments/"
    if not os.path.isdir(base_path):
        os.mkdir(base_path)
    
    experiment_number = len(os.listdir(base_path))

    experiment_path = base_path + f"experiment_00{experiment_number+1}/"
    
    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)

    generator_path = experiment_path + "generator.pth"
    discriminator_path = experiment_path + "discriminator.pth"

    image_path = experiment_path + "images/"
    if not os.path.isdir(image_path):
        os.mkdir(image_path)


    generator = Generator(z_dim).to(device)
    discriminator = Discriminator(1).to(device)
    initialize_weights(generator)
    initialize_weights(discriminator)

    gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    criterion = nn.BCELoss()


    generator.train()
    discriminator.train()

    for epoch in range(epochs):

        if epoch % 10 == 0 and epoch > 0:
            saveModel(epoch, 
            generator.state_dict(), gen_optimizer.state_dict(), generator_path,
            discriminator.state_dict(), dis_optimizer.state_dict(), discriminator_path)
        
        for i, data in enumerate(loader, 0):
            image = data
            image = image.to(device)
            batch_size = image.shape[0]

            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = generator(noise)


            dis_real = discriminator(image).reshape(-1) # N
            loss_dis_real = criterion(dis_real, torch.ones_like(dis_real))
            dis_fake = discriminator(fake.detach()).reshape(-1)
            loss_dis_fake = criterion(dis_fake, torch.zeros_like(dis_fake))

            loss_dis = (loss_dis_real + loss_dis_fake) / 2
            discriminator.zero_grad()
            loss_dis.backward()
            dis_optimizer.step()
                        

            gen_fake = discriminator(fake).reshape(-1)
            loss_gen = criterion(gen_fake, torch.ones_like(gen_fake))
            generator.zero_grad()
            loss_gen.backward()
            gen_optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}/{epochs}")
            saveImages(fixed_noise, generator, image_path, epoch)
    
    saveModel(epoch, 
            generator.state_dict(), gen_optimizer.state_dict(), generator_path,
            discriminator.state_dict(), dis_optimizer.state_dict(), discriminator_path)
    

    # compute FID 5 times and take avg value
    distances = []
    for _ in range(5):
        distance = fidEval(fid, generator, loader, z_dim)
        distances.append(distance)
    
    eval_score = sum(distances) / len(distances)

    print(f"FID for this experiment is: {eval_score}")

    descriptions = {
        "type": "DCGAN",
        "epochs": epochs,
        "z_dim": z_dim,
        "learning_rate": learning_rate,
        "experiment_path": experiment_path,
        "FID": eval_score
    }

    saveToCsv(descriptions)



if __name__=="__main__":
    
    seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS = 1
    Z_DIM = 500
    LEARNING_RATE = 1e-4
    FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(device)
    CSV_PATH = "PATH_TO_CSV"

    dataset = LungsDataset(CSV_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    fid = MyFID(device)
    

    trainModel(device, LEARNING_RATE, FIXED_NOISE, EPOCHS, Z_DIM, fid)