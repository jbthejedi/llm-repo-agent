from together import Together
from dotenv import load_dotenv

def main():
  load_dotenv()
  client = Together()

  response = client.endpoints.create(
      model="justinbarrye_c241/Qwen3-8B-qwen3-8b-sft-pilot-798551d2",
      display_name="justinbarrye_c241/Qwen3-8B-qwen3-8b-sft-pilot-798551d2",
      hardware="1x_nvidia_h100_80gb_sxm",
      min_replicas=1,
      max_replicas=1
  )
  print(response)

if __name__ == '__main__':
  main()