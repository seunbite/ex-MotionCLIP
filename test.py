import sys
sys.path.append('.')
import os
import torch
import numpy as np
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.utils.misc import load_model_wo_clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.tensors import collate
import clip
from src.visualize.visualize import get_gpu_device
from src.utils.action_label_to_idx import action_label_to_idx
import glob

class MotionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path="/scratch2/iyy1112/motion-persona/save/20250805_mdm_type3/*/*.npy"):
        self.data_files = glob.glob(data_path)
        print(f"Found {len(self.data_files)} motion files")
        if len(self.data_files) == 0:
            raise ValueError(f"No files found matching pattern: {data_path}")
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index):
        # Load .npy file
        data = np.load(self.data_files[index], allow_pickle=True).item()
        
        # Extract motion data (3D tensor)
        motion = data['motion']  # Shape: [frames, joints, features]
        
        # Convert to torch tensor and ensure correct format
        motion_tensor = torch.from_numpy(motion).float()
        
        # Reshape to match expected format: [joints, features, frames]
        if len(motion_tensor.shape) == 3:
            motion_tensor = motion_tensor.permute(1, 2, 0)  # [joints, features, frames]
        
        # Pad or truncate to 100 frames if necessary
        target_frames = 100
        current_frames = motion_tensor.shape[-1]
        
        if current_frames > target_frames:
            # Truncate to 100 frames
            motion_tensor = motion_tensor[:, :, :target_frames]
        elif current_frames < target_frames:
            # Pad with last frame
            padding = motion_tensor[:, :, -1:].repeat(1, 1, target_frames - current_frames)
            motion_tensor = torch.cat([motion_tensor, padding], dim=-1)
        
        # Create dummy text label (you can modify this based on your needs)
        dummy_text = "walking"  # Default action
        
        return {
            'inp': motion_tensor,
            'target': 0,  # Dummy target
            'clip_text': dummy_text,
            'all_categories': [dummy_text]
        }

if __name__ == '__main__':
    import argparse
    
    # Create argument parser
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("checkpointname", help="Path to the checkpoint file")
    parser_arg.add_argument("--data_path", default="/scratch2/iyy1112/motion-persona/save/20250805_mdm_type3/*/*.npy", 
                           help="Path pattern for motion data files")
    args = parser_arg.parse_args()
    
    parameters, folder, checkpointname, epoch = parser(checkpoint=True)
    gpu_device = get_gpu_device()
    parameters["device"] = f"cuda:{gpu_device}"
    parameters['use_action_cat_as_text_labels'] = True
    parameters['only_60_classes'] = True

    TOP_K_METRIC = 5

    # Load model and create dataset
    model, _ = get_model_and_data(parameters, split='vald')
    
    # Create custom dataset with specified data path
    dataset = MotionDataset(data_path=args.data_path)
    
    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)
    model.eval()

    iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
                          shuffle=False, num_workers=8, collate_fn=collate)

    action_text_labels = list(action_label_to_idx.keys())
    action_text_labels.sort(key=lambda x: action_label_to_idx[x])

    texts = clip.tokenize(action_text_labels[:60]).to(model.device)
    classes_text_emb = model.clip_model.encode_text(texts).float()

    correct_preds_top_5, correct_preds_top_1 = 0, 0
    total_samples = 0
    
    # Store predictions for analysis
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            if isinstance(batch['x'], list):
                continue
            for key in batch.keys():
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(parameters['device'])
            
            batch = model(batch)
            texts = clip.tokenize(batch['clip_text']).to(model.device)
            batch['clip_text_embed'] = model.clip_model.encode_text(texts).float()
            
            # For custom dataset, we don't have ground truth labels, so we'll just compute similarities
            classes_text_emb_norm = classes_text_emb / classes_text_emb.norm(dim=-1, keepdim=True)
            motion_features_norm = batch['z'] / batch['z'].norm(dim=-1, keepdim=True)
            scores = motion_features_norm @ classes_text_emb_norm.t()
            similarity = (100.0 * motion_features_norm @ classes_text_emb_norm.t()).softmax(dim=-1)

            total_samples += similarity.shape[0]
            
            # Get top predictions for each sample
            for i in range(similarity.shape[0]):
                values, indices = similarity[i].topk(5)
                
                # Store top-5 predictions
                top_5_predictions = [action_text_labels[idx] for idx in indices]
                top_1_prediction = action_text_labels[indices[0]]
                
                all_predictions.append({
                    'top_1': top_1_prediction,
                    'top_5': top_5_predictions,
                    'scores': values.cpu().numpy()
                })
                
                print(f"Sample {total_samples + i}:")
                print(f"  Top-1: {top_1_prediction}")
                print(f"  Top-5: {top_5_predictions}")
                print(f"  Scores: {values.cpu().numpy()}")
                print("---")

        print(f"Processed {total_samples} motion samples")
        print("Action classification completed!")
        
        # Save predictions to file
        import json
        with open('motion_predictions.json', 'w') as f:
            json.dump(all_predictions, f, indent=2)
        print("Predictions saved to motion_predictions.json") 