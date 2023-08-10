from    nltk.translate.bleu_score   import  corpus_bleu, SmoothingFunction

def calculate_bleu_score(reference, hypothesis):
    smoothie = SmoothingFunction().method4  # Smoothing method for BLEU score calculation
    # Convert reference and hypothesis sequences to lists of strings
    reference = [[str(token) for token in ref] for ref in reference]
    hypothesis = [[str(token) for token in hyp] for hyp in hypothesis]
    # Calculate BLEU score
    return corpus_bleu(reference, hypothesis, smoothing_function=smoothie)