from llava.train.train import train_spatial

if __name__ == "__main__":
    train_spatial(attn_implementation="flash_attention_2")
