import hydra
from omegaconf import DictConfig

from dataloader import TrecDLLoader, TrecRobust04Loader

@hydra.main(version_base=None, config_path=".", config_name=None)
def main(cfg):
    dataloader_type = cfg.dataloader.dataloader_type
    dataloader_config = cfg.dataloader

    if dataloader_type == "trec-dl":
        dataloader = TrecDLLoader(dataloader_config)
    elif dataloader_type == "robust04":
        dataloader = TrecRobust04Loader(dataloader_config)
    else:
        print(f"Dataloader {dataloader_type} is not known.")
    
    print("Loading document data")
    documents = dataloader.get_documents()
    queries = dataloader.get_queries()
    user_input = input("Please type the number of chars to display: ")
    max_chars = int(user_input)
    print("Waiting for user input...")
    while True:
        result = wait_for_id()
        if result.lower() == 'exit':
            print("Exiting the program...")
            break
        if result[:2] == 'q ':
            result = result[2:]
            if result not in queries:
                print("No query with id", result)
                continue
            text = queries[result]
            text = text.replace("\n", " ")
            print("Query:")
            print(text[:max_chars])
        else:
            if result not in documents:
                print("No document with id", result)
                continue
            text = documents[result]
            text = text.replace("\n", " ")
            print("Document:")
            print(text[:max_chars])
        print()
    

def wait_for_id():
    user_input = input("Please type the id (or type 'exit' to quit). Format is 'q <query_id>' or '<doc_id>': ")
    return user_input

if __name__ == "__main__":
    main()
