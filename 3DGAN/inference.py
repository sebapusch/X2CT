import torch
import numpy as np
import nibabel as nib
import os
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model

def save_ct_to_nifti(ct_volume, output_path):
    # Assumes ct_volume is a numpy array with shape (1, D, H, W)
    ct_volume = np.squeeze(ct_volume, axis=0)  # remove channel dim if present
    nifti_img = nib.Nifti1Image(ct_volume.astype(np.float32), affine=np.eye(4))
    nib.save(nifti_img, output_path)

def run_inference(
        yml_path,
        output_dir,
        data_root,
        dataset_file,
):
    # Load configuration
    cfg_from_yaml(yml_path)
    opt = merge_dict_and_yaml({
        'dataset_class': 'align_ct_xray_std',
        'tag': 'd2_multiview2500',
        'data': 'LIDC256',
        'model_class': 'MultiViewCTGAN',
        'dataset_file': dataset_file,
        'dataroot': data_root,
    }, cfg)

    # Device
    opt.gpu_ids = [0] if torch.cuda.is_available() else []
    opt.serial_batches = True

    # Load dataset
    datasetClass, _, _, collateClass = get_dataset(opt.dataset_class)
    dataset = datasetClass(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collateClass)

    # Load model
    model = get_model(opt.model_class)()
    model.eval()
    model.init_process(opt)
    model.setup(opt)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Starting inference and saving CT volumes...")
    for i, data in enumerate(dataloader):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        generated_ct = visuals['G_fake'].data.cpu().numpy()

        # Save to NIfTI file
        file_name = f"reconstructed_{i:04d}.nii.gz"
        file_path = os.path.join(output_dir, file_name)
        save_ct_to_nifti(generated_ct, file_path)
        print(f"Saved {file_path}")

    print("Inference completed.")

if __name__ == '__main__':
    run_inference(
        yml_path='./experiment/multiview2500/d2_multiview2500.yml',  # replace with your config file
        output_dir='./data/experiment_outputs',
        dataset_file='./data/sample.txt',
        data_root='./data/experiment',
    )
