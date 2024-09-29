from diffusers import FluxPipeline
import torch

def test_prepare_latent_image_ids():
    # Set up test parameters
    batch_size = 2
    height = 8
    width = 8
    device = 'cpu'  # Use CPU for easy printing
    dtype = torch.float32

    # Call the function
    result = FluxPipeline._prepare_latent_image_ids(batch_size, height, width, device, dtype)

    print(f"Result shape: {result.shape}")
    print("\nFirst batch item:")
    for i in range(16):  # 16 = (height // 2) * (width // 2)
        print(f"Pixel {i}: {result[0, i].tolist()}")

    print("\nSecond batch item:")
    for i in range(16):
        print(f"Pixel {i}: {result[1, i].tolist()}")

    print("\nFull result:")
    print(result)

# Run the test
test_prepare_latent_image_ids()