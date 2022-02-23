import pysbd,math
seg = pysbd.Segmenter(language="en", clean=False)

def extract_snippets(text:str,n:int=2):
        """ 
        Extracts snippets from text with a sliding window 
        n : sentences per snippet returns list of sentences
        returns list of sentences
        """
        sentences = seg.segment(text)
        snippets = []
        i = 0
        last_index = 0
        while i < len(sentences):
            snippet = ' '.join(sentences[i:i + n])
            if len(snippet.split(' ')) > 4:
                snippets.append(snippet)
            last_index = i + n
            i += int(math.ceil(n / 2))
        if last_index < len(sentences):
            snippet = ' '.join(sentences[last_index:])
            if len(snippet.split(' ')) > 4:
                snippets.append(snippet)
        return snippets
