import pandas as pd
import six

import py_stringmatching as sm
import py_entitymatching.utils.generic_helper as gh



# q-gram tokenizer
def tok_qgram(input_string, q):
    """
    This function splits the input string into a list of q-grams. Note that,
    by default the input strings are padded and then tokenized.
    Args:
        input_string (string): Input string that should be tokenized.
        q (int): q-val that should be used to tokenize the input string.
    Returns:
        A list of tokens, if the input string is not NaN,
        else returns NaN.
    Examples:
        >>> import py_entitymatching as em
        >>> em.tok_qgram('database', q=2)
        ['#d', 'da', 'at', 'ta', 'ab', 'ba', 'as', 'se', 'e$']
        >>> em.tok_qgram('database', q=3)
        ['##d', '#da', 'dat', 'ata', 'tab', 'aba', 'bas', 'ase', 'se$', 'e$$']
        >>> em.tok_qgram(None, q=2)
        nan
    """

    if pd.isnull(input_string):
        return pd.np.NaN

    input_string = gh.convert_to_str_unicode(input_string)
    measure = sm.QgramTokenizer(qval=q)

    return measure.tokenize(input_string)


def tok_delim(input_string, d):
    """
    This function splits the input string into a list of tokens
    (based on the delimiter).
    Args:
        input_string (string): Input string that should be tokenized.
        d (string): Delimiter string.
    Returns:
        A list of tokens, if the input string is not NaN ,
        else returns NaN.
    Examples:
        >>> import py_entitymatching as em
        >>> em.tok_delim('data science', ' ')
        ['data', 'science']
        >>> em.tok_delim('data$#$science', '$#$')
        ['data', 'science']
        >>> em.tok_delim(None, ' ')
        nan
    """

    if pd.isnull(input_string):
        return pd.np.NaN

    input_string = gh.convert_to_str_unicode(input_string)

    measure = sm.DelimiterTokenizer(delim_set=[d])

    return measure.tokenize(input_string)