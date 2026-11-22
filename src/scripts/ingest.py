import pymupdf
import argparse
import hashlib
from pathlib import Path
from medical_chatbot.knowledge import ensure_index, embed

#Splitting the text into chuncks 
def chunks(text, size=1100, overlap=180):
    text= ' '.join(text.split())
    step = size - overlap
    result = []
    for i in range(0, len(text), step):
        chunk = text[i:i+size]
        if len(chunk) > 0:
            result.append(chunk)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', default=r'D:\Tanmoy\Python_Projects_Self_learn\Medical_Chatbot_Project(Agentic_AI)\medical-chatbot\src\data\raw_data\Medical_book.pdf')
    args = parser.parse_args()
    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f'PDF not found {pdf}')
    index = ensure_index()
    
    with pymupdf.open(pdf) as doc:
        
        batch = []
        
        for page_number, page in enumerate(doc, start=1):
            
            for part, content in enumerate(chunks(page.get_text()), start=1):
                
                key = hashlib.sha256(
                    f'{pdf.name}:{page_number}:{part}:{content}'.encode()
                ).hexdigest()
                
                batch.append((
                    key, content, {
                        'page' : page_number,
                        'text' : content,
                        'file' : pdf.name
                    }
                ))
                
                if len(batch) >=50:
                    vectors = embed([b[1] for b in batch])
                    index.upsert(
                        vectors=[
                            {
                                'id': b[0], 
                                'values': v, 
                                'metadata':b[2]
                            }
                            for b, v in zip(batch, vectors) 
                        ]
                    )  
                    batch=[]     
            
            if page_number % 50 == 0:
                print(
                    f'Indexed pages: {page_number}/{len(doc)}',
                      flush=True
                    )
        
        if batch:
            vectors = embed([b[1] for b in batch])
            index.upsert(
                vectors= [
                    {
                        'id' : b[0],
                        'values': v,
                        'metadata': b[2]
                    }
                    for b,v in zip(batch, vectors)
                ]
            )              
    print("--------------Ingestion Complete--------------")
    
if __name__ == "__main__":
    main()
            
            