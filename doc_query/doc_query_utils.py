import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from PyPDF2 import PdfReader
import tiktoken
import pickle

# extract all the text from a pdf file
def extract_text_frfom_pdf(doc, npages=None):
    reader = PdfReader(doc)

    n = len(reader.pages)
    if npages is not None:
        npages = min(npages, n)
    else:
        npages = n

    print(f'You have {n:,} page(s) in your file, loading {npages:,}')

    text = ''
    for i in range(npages):
        text += reader.pages[i].extract_text()

    return text


def partition_text(full_text, psize=500, delim='\n'):
    text_chunks = full_text.split(delim)

    ptext = []
    next_item = ''
    next_item_size = 0

    for i in range(len(text_chunks)):
        txt = text_chunks[i]

        next_item_size += len(txt)
        next_item += txt + ' '

        if next_item_size > psize:
            next_item_size = 0
            ptext.append(next_item)
            next_item = ''

    return ptext


def create_overlapped_partitions(ptext, overlap_pct=0.5, delim=' '):
    assert len(ptext) > 1, 'Insufficient text to partition'

    opart = []
    pos1 = int(float(len(ptext[0]) * overlap_pct))

    for i in range(1, len(ptext)):
        pos2 = int(float(len(ptext[i]) * overlap_pct))

        while ptext[i - 1][pos1] != delim:
            pos1 += 1
        while ptext[i][pos2] != delim:
            pos2 += 1

        new_part = ptext[i - 1][pos1:]
        new_part += ptext[i][:pos2]
        opart.append(new_part)

        pos1 = pos2

    return opart

# split text into partitions, including overlapping partitions
def create_text_partitions(full_text, psize=500, overlap_pct=0.5):
    ptext = partition_text(full_text, psize=psize)
    optext = create_overlapped_partitions(ptext, overlap_pct)
    lp = len(ptext)
    lop = len(optext)
    ptext.extend(optext)
    nc = len(ptext)
    print(f'Loaded {nc:,} chuncks : {lp:,} chuncks and {lop:,} overlap chunks')
    return ptext

def get_default_embedding_name(cfg):
    return 'emb_' + cfg['PDF_DOC'].split('.')[0]+'.pkl'

def load_embeddings(cfg, fname=None):
    if fname == None:
        fname = cfg['DATA_FOLDER'] + "\\" + get_default_embedding_name(cfg)

    print(fname)

    with open(fname, 'rb') as f:
        emb_info = pickle.load(f)

    sl = [len(x[2]) for x in emb_info]
    assert max(sl) == min(sl), 'incompatible embedding sizes'
    print(f'Loaded {len(sl):,} embeddings, each of size {max(sl):,}')
    return emb_info

# convert a list of texts into embedding vectors
def create_embeddings(cfg, load_saved=True, fname=None):
    if load_saved:
        return load_embeddings(cfg, fname)

    text = extract_text_frfom_pdf(cfg['DATA_FOLDER'] + "\\" + cfg['PDF_DOC'])
    ptext = create_text_partitions(text,
                                   psize=cfg['TEXT_PARTITION_SIZE'],
                                   overlap_pct=cfg['OVERLAP_PCT'])

    emb_info = []
    encoding = tiktoken.get_encoding(cfg['EMBEDDING_ENCODING'])

    if fname == None:
        fname = cfg['DATA_FOLDER'] + "\\" + get_default_embedding_name(cfg)

    for i in range(len(ptext)):
        txt = ptext[i]
        ntokens = len(encoding.encode(txt))
        emb = get_embedding(txt, engine=cfg['EMBEDDING_MODEL'])

        emb_info.append([txt, ntokens, emb])

    if load_saved == False:
        with open(fname, 'wb') as f:
            pickle.dump(emb_info, f)

    return emb_info

# compare the embedding vector of a query with the
# embedding vectors corresponding to document chuncks
def find_similar_text(query, emb, cfg, sim_threshold=0.8):
    qe = get_embedding(query, engine=cfg['EMBEDDING_MODEL'])
    sim = [cosine_similarity(e[2], qe) for e in emb]

    res_info = []
    sim_text = ''
    for i in range(len(emb)):
        if sim[i] >= sim_threshold:
            res_info.append([emb[i][0], emb[i][1], emb[i][2], sim[i]])
            sim_text += emb[i][0] + ' '

    return sim_text, res_info


# there is a whole "prompt engineering" field
# this is a super simple way to create a gpt prompt
def get_prompt(question, doc_specific=True, doc_text=''):
    if doc_specific:
        prompt = 'The document provided contains the following information: ' \
                 + doc_text + ' ' + question
        role_descr = 'You answer questions about the document provided. If the information is not in the document say you do not know the answer.'
    else:
        prompt = question
        role_descr = 'You answer the question asked.'

    return prompt, role_descr


# this is the main function
def ask_question(question,  # text of the question
                 doc_embeddings,  # the embeddings of the PDF
                 cfg,  # configuration settings
                 doc_specific=True,  # makes the answer doc specific only
                 verbose=True,
                 sim_threshold=0.8  # threshold to determine what embeddings to use in the prompt
                 ):
    similar_text, sim_info = find_similar_text(question,
                                               doc_embeddings,
                                               cfg,
                                               sim_threshold)

    prompt, role_descr = get_prompt(question, doc_specific, similar_text)

    messages = [
        {"role": "system", "content": role_descr},
        {"role": "user", "content": prompt},
    ]

    response_info = openai.ChatCompletion.create(
        model=cfg['GPT_MODEL'],
        messages=messages,
        temperature=0
    )

    response = response_info['choices'][0]['message']['content']

    if verbose:
        print(f'Answer: {response}')

    return response, response_info

