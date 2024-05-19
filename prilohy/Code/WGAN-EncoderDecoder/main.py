import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
from torchvision.io import read_image
from torchvision.utils import save_image, make_grid
import random
import os
import pandas as pd

import datetime

import sys
FOLDER_PATH = "\\".join(os.getcwd().split('\\')[:-1])
sys.path.append(FOLDER_PATH)
from FID.FID import MyFID

from Generator import Generator
from Discriminator import Critic
from Dataset import LungsDataset
from Utils import seed_everything, initialize_weights, gradient_penalty


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

def saveImages(generator, image_path, epoch, device, fixed_masks, fixed_noise):
    print("Saving images.")        
    with torch.no_grad():
        fake = generator(fixed_masks, fixed_noise)
        
        img_grid_fake = torchvision.utils.make_grid(fake[:], normalize=True)
        mask_grid = torchvision.utils.make_grid(fixed_masks[:], normalize=True)

        file_path_image = image_path + f"Fake_image-{epoch}.png"
        torchvision.utils.save_image(img_grid_fake, file_path_image)

        file_path_mask = image_path + f"Fake_image-{epoch}-mask.png"
        torchvision.utils.save_image(mask_grid, file_path_mask)

def fidEval(fid, generator, loader, z_dim):
    distances = []
    for _, data in enumerate(loader, 0):
        real_images = data[0]
        masks = data[1]

        real_images = real_images.to(device)
        
        masks = masks.to(device)
        batch_size = masks.shape[0]

        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_images = generator(masks, noise)

        distance = fid.get_distance(real_images, fake_images)

        distances.append(distance)
    
    final_distance = sum(distances) / len(distances)

    return final_distance

def saveToCsv(descriptions):
    csv_path = "./experiments/experimentsInfo.csv"
    if not os.path.isfile(csv_path):
        df = pd.DataFrame([descriptions])
    else:
        df = pd.read_csv(csv_path, index_col=0)
        
        df_e = pd.DataFrame([descriptions])
        df = pd.concat([df, df_e])
    
    df.to_csv(csv_path, index_label=False)

def trainModel(device, learning_rate, epochs, critic_iterations, lambda_gp, z_dim, fixed_masks, fixed_noise, fid):

    # create folder where specific experiment will be saved
    base_path = "./experiments/"
    if not os.path.isdir(base_path):
        os.mkdir(base_path)
    experiment_number = len(os.listdir(base_path))
    
    experiment_path = base_path + f"experiment_00{experiment_number+1}/"
    
    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)

    generator_path = experiment_path + "generator.pth"
    critic_path = experiment_path + "critic.pth"

    image_path = experiment_path + "images/"
    if not os.path.isdir(image_path):
        os.mkdir(image_path)


    generator = Generator(1).to(device)
    critic = Critic(2).to(device)
    initialize_weights(generator)
    initialize_weights(critic)

    gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))


    generator.train()
    critic.train()

    for epoch in range(epochs):

        if epoch % 10 == 0 and epoch > 0:
            saveModel(epoch, 
            generator.state_dict(), gen_optimizer.state_dict(), generator_path,
            critic.state_dict(), critic_optimizer.state_dict(), critic_path)
        
        for i, data in enumerate(loader, 0):
            image = data[0]
            mask = data[1]
        
            image = image.to(device)
            mask = mask.to(device)
            batch_size = image.shape[0]

            for _ in range(critic_iterations):
                noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake = generator(mask, noise)

                ## Train Discriminator
                critic_real = critic(image, mask).reshape(-1)
                critic_fake = critic(fake, mask).reshape(-1)
            
                gp = gradient_penalty(critic, mask, image, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
                    )
                        
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                critic_optimizer.step()


            ## Train Generator
            gen_fake = critic(fake, mask).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            generator.zero_grad()
            loss_gen.backward()
            gen_optimizer.step()


        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}]")
            saveImages(generator, image_path, epoch, device, fixed_masks, fixed_noise)
    
    saveModel(epoch, 
            generator.state_dict(), gen_optimizer.state_dict(), generator_path,
            critic.state_dict(), critic_optimizer.state_dict(), critic_path)
    
    distances = []
    for _ in range(5):
        distance = fidEval(fid, generator, loader, z_dim)
        distances.append(distance)
    
    eval_score = sum(distances) / len(distances)
    
    print(f"FID for this experiment is: {eval_score}")
    
    # descriptions
    descriptions = {
        "type": 'WGAN-EncoderDecoder',
        "epochs": epochs,
        "z_dim": z_dim,
        "learning_rate": learning_rate,
        "critic_iterations": critic_iterations,
        "lambda_gp": lambda_gp,
        "experiment_path": experiment_path,
        "FID": eval_score
    }

    saveToCsv(descriptions)



if __name__=="__main__":
    
    seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LungsDataset(CSV_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    fid = MyFID(device)
   
    BATCH_SIZE = 16
    EPOCHS = 1
    Z_DIM = 500
    LAMBDA_GP = 10
    LEARNING_RATE = 1e-4
    CRITTIC_ITERATIONS = 5
    CSV_PATH = "PATH_TO_CSV"
    FIXED_MASKS = dataset.getEvalMask(4).to(device)
    FIXED_NOISE = torch.randn(4, Z_DIM, 1, 1).to(device)

    trainModel(device, LEARNING_RATE, EPOCHS, CRITTIC_ITERATIONS, LAMBDA_GP, Z_DIM, FIXED_MASKS, FIXED_NOISE, fid)