import pathlib
import time
import torch
import os
import pandas as pd

from diffusers import StableDiffusionPipeline
from utils import logger, visual_utils
import torchvision.utils
import torchvision.transforms.functional as tf

def make_QBench():
    prompts = ["A bus", #0
               "A bus and a bench", #1
               "A bus next to a bench and a bird", #2
               "A bus next to a bench with a bird and a pizza", #3
               "A green bus", #4
               "A green bus and a red bench", #5
               "A green bus next to a red bench and a pink bird", #6
               "A green bus next to a red bench with a pink bird and a yellow pizza", #7
               "A bus on the left of a bench", #8
               "A bus on the left of a bench and a bird", #9
               "A bus and a pizza on the left of a bench and a bird", #10
               "A bus and a pizza on the left of a bench and below a bird", #11
               ]
    
    ids = []

    for i in range(len(prompts)):
        ids.append(str(i).zfill(3))

    bboxes = [[[2,121,251,460]],#0
            [[2,121,251,460], [274,345,503,496]],#1
            [[2,121,251,460], [274,345,503,496],[344,32,500,187]],#2
            [[2,121,251,460], [274,345,503,496],[344,32,500,187],[58,327,187,403]],#3
            [[2,121,251,460]],#4
            [[2,121,251,460], [274,345,503,496]],#5
            [[2,121,251,460], [274,345,503,496],[344,32,500,187]],#6
            [[2,121,251,460], [274,345,503,496],[344,32,500,187],[58,327,187,403]],#7
            [[2,121,251,460],[274,345,503,496]],#8
            [[2,121,251,460],[274,345,503,496],[344,32,500,187]],#9
            [[2,121,251,460], [58,327,187,403], [274,345,503,496],[344,32,500,187]],#10
            [[2,121,251,460], [58,327,187,403], [274,345,503,496],[344,32,500,187]],#11
            ]

    phrases = [["bus"],#0
               ["bus", "bench"],#1
               ["bus", "bench", "bird"],#2
               ["bus","bench","bird","pizza"],#3
               ["bus"],#4
               ["bus", "bench"],#5
               ["bus", "bench", "bird"],#6
               ["bus","bench","bird","pizza"],#7
               ["bus","bench"],#8
               ["bus","bench","bird"],#9
               ["bus","pizza","bench","bird"],#11
               ["bus","pizza","bench","bird"]#12
               ]

    token_indices = [[2],#0
                     [2,5],#1
                     [2, 6, 9],#2
                     [2,6,9,12],#3
                     [3],#4
                     [3,7],#5
                     [3,8,12],#6
                     [3,8,12,16],#7
                     [2,8],#8
                     [2,8,11],#9
                     [2,5,11,14],#10
                     [2,5,11,15],#11
                     ]
    data_dict = {
    i: {
        "id": ids[i],
        "prompt": prompts[i],
        "bboxes": bboxes[i],
        "phrases": phrases[i],
        "token_indices": token_indices[i]
    }
    for i in range(len(prompts))
    }
    return data_dict

def readPromptsCSV(path):
    df = pd.read_csv(path, dtype={'id': str})
    conversion_dict={}
    for i in range(0,len(df)):
        conversion_dict[df.at[i,'id']] = {
            'prompt': df.at[i,'prompt'],
            'obj1': df.at[i,'obj1'],
            'bbox1':df.at[i,'bbox1'],
            'obj2': df.at[i,'obj2'],
            'bbox2':df.at[i,'bbox2'],
            'obj3': df.at[i,'obj3'],
            'bbox3':df.at[i,'bbox3'],
            'obj4': df.at[i,'obj4'],
            'bbox4':df.at[i,'bbox4'],
        }
    
    return conversion_dict    

def samples():
    prompts = ["a scene of a busy marketplace in a medieval town",
               "an illustration of a vibrant city park during peak bloom season",
               "a serene forest scene with various wildlife present.",
               "A black dog running on the beach with a white dog wearing a red collar"
               ]
    
    data_dict = {
    i: {
        "prompt": prompts[i]
    }
    for i in range(len(prompts))
    }
    return data_dict

# Generate an image described by the prompt and
# insert objects described by text at the region defined by bounding boxes
def main():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",use_safetensors=False, safety_checker=None)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    pipe = pipe.to(device)

    height=512
    width=512
    seeds = range(1,17)

    #bench=make_tinyHRS()
    bench=readPromptsCSV(os.path.join("prompts","prompt_collection_bboxes.csv"))

    model_name="PromptCollection-SD14"
    
    if (not os.path.isdir("./results/"+model_name)):
            os.makedirs("./results/"+model_name)
    
    #intialize logger
    l=logger.Logger("./results/"+model_name+"/")
    
    # ids to iterate the dict
    ids = []
    for i in range(0,len(bench)):
        ids.append(str(i).zfill(3))
        
    for id in ids:
        
        output_path = "./results/"+model_name+"/"+ id +'_'+bench[id]['prompt'] + "/"

        if (not os.path.isdir(output_path)):
            os.makedirs(output_path)

        print("Sample number ",id)
        
        torch.cuda.empty_cache()

        gen_images=[]
        gen_bboxes_images=[]

        for seed in seeds:
            print(f"Current seed is : {seed}")

            # start stopwatch
            start = time.time()

            if torch.cuda.is_available():
                g = torch.Generator('cuda').manual_seed(seed)
            else:
                g = torch.Generator('cpu').manual_seed(seed)

            images = pipe(
                prompt=bench[id]['prompt'],
                height=height,
                width=width,
                output_type="pil",
                num_inference_steps=50,
                generator=g,
                negative_prompt='low quality, low res, distortion, watermark, monochrome, cropped, mutation, bad anatomy, collage, border, tiled').images

            # end stopwatch
            end = time.time()
            # save to logger
            l.log_time_run(start, end)

            image=images[0]

            image.save(output_path +"/"+ str(seed) + ".jpg")
            gen_images.append(tf.pil_to_tensor(image))

        # save a grid of results across all seeds without bboxes
        tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_images,nrow=4,padding=0)).save(output_path +"/"+ bench[id]['prompt'] + ".png")
    
    # log gpu stats
    l.log_gpu_memory_instance()
    # save to csv_file
    l.save_log_to_csv(model_name)
        
if __name__ == '__main__':
    main()
