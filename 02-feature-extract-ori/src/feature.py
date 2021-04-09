import py_stringmatching as sm
import py_entitymatching.utils.generic_helper as gh
import pandas as pd
from data_type import get_type
import tokenizer
import sys
from time import time

def feature_header_init(data, attr_pair):
	string_functions = {
						"aff": affine, "levd": lev_dist, "levs": lev_sim, 
						"jar": jaro, 
						"jwn": jaro_winkler, 
						"nmw": needleman_wunsch, 
						"sw": smith_waterman, "exm": exact_match, "mel": monge_elkan}
	
	arr_functions = {"ocf": overlap_coeff, "jac": jaccard, "dice": dice, 
					 "cos": cosine}

	num_functions = {"anm": abs_norm, "rdf": rel_diff}

	header = []
	header.append('label')
	for pair in attr_pair:
		d_type_left = get_type(data[pair[0]])
		d_type_right = get_type(data[pair[1]])

		# apply all the string and arr functions because we can alwasy treat text as strings
		for name, sim_func in string_functions.items():
			header.append('str' + '__' + name + '__' + pair[0].replace('ltable_', '') + '__' + pair[1].replace('rtable_', ''))
		
		for name, sim_func in arr_functions.items():
			header.append('arr' + '__' + name + '__' + '3gram' + '__' + pair[0].replace('ltable_', '') + '__' + pair[1].replace('rtable_', ''))
			header.append('arr' + '__' +  name + '__' + 'space' + '__' + pair[0].replace('ltable_', '') + '__' + pair[1].replace('rtable_', ''))
	
		if d_type_left == 'numeric' and d_type_right == 'numeric':
			for name, sim_func in num_functions.items():
				header.append('num' + '__' + name  + '__' + pair[0].replace('ltable_', '') + '__' + pair[1].replace('rtable_', ''))
	#print(header)
	return header	


def get_feature_beta(data, header, time_list):
	string_functions = {
						"aff": affine, "levd": lev_dist, "levs": lev_sim, 
						"jar": jaro, 
						"jwn": jaro_winkler, 
						"nmw": needleman_wunsch, 
						"sw": smith_waterman, "exm": exact_match, "mel": monge_elkan}
	
	arr_functions = {"ocf": overlap_coeff, "jac": jaccard, "dice": dice, 
					 "cos": cosine}

	num_functions = {"anm": abs_norm, "rdf": rel_diff}

	feature_dict = {}
	new_header = []

	for col_name in header: 
		elements = col_name.split('__')
		start = time()
		feature_dict[col_name] = []

		if col_name == 'label':
			feature_dict[col_name] = []
			for index, row in data.iterrows():
				feature_dict[col_name].append(row[col_name])

		if elements[0] == 'str':
			col_one_eles = data['ltable_'+ elements[2]].tolist()
			col_two_eles = data['rtable_' + elements[3]].tolist()
			func_name = elements[1]

			if len(col_one_eles) != len(col_two_eles):
				print("error")
			for i in range(0, len(col_one_eles)):
				s1 = str(col_one_eles[i])
				s2 = str(col_two_eles[i])
				#s1 = unicode(s1, errors='ignore')
				#s2 = unicode(s2, errors='ignore')
				feature_dict[col_name].append(string_functions[func_name](s1, s2))

		elif elements[0] == 'arr':
			col_one_eles = data['ltable_' + elements[3]].tolist()
			col_two_eles = data['rtable_' + elements[4]].tolist()
			func_name = elements[1]
			delimiter = elements[2]

			if len(col_one_eles) != len(col_two_eles):
				print("error")

			if delimiter == '3gram':
				for i in range(0, len(col_one_eles)):
					s1 = str(col_one_eles[i])
					s2 = str(col_two_eles[i])
					#s1 = unicode(s1, errors='ignore')
					#s2 = unicode(s2, errors='ignore')
					l1 = tokenizer.tok_qgram(s1, 3)
					l2 = tokenizer.tok_qgram(s2, 3)
					feature_dict[col_name].append(arr_functions[func_name](l1, l2))
			elif delimiter == 'space':
				for i in range(0, len(col_one_eles)):
					s1 = str(col_one_eles[i])
					s2 = str(col_two_eles[i])
					l1 = tokenizer.tok_delim(s1, ' ')
					l2 = tokenizer.tok_delim(s2, ' ')
					feature_dict[col_name].append(arr_functions[func_name](l1, l2))

		elif elements[0] == 'num':
			col_one_eles = data['ltable_' + elements[2]].tolist()
			col_two_eles = data['rtable_' + elements[3]].tolist()
			func_name = elements[1]

			if len(col_one_eles) != len(col_two_eles):
				print("error")
			for i in range(0, len(col_one_eles)):
				s1 = str(col_one_eles[i])
				s2 = str(col_two_eles[i])
				#s1 = unicode(s1, errors='ignore')
				#s2 = unicode(s2, errors='ignore')
				feature_dict[col_name].append(num_functions[func_name](s1, s2))

		if col_name not in time_list:
			time_list[col_name] = 0
		time_list[col_name] += time() - start
		feature_vec = pd.DataFrame(data=feature_dict)
	
	return feature_vec


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

## String based similarity measures
def affine(s1, s2):
    """
    This function computes the affine measure between the two input strings.

    Args:
        s1,s2 (string ): The input strings for which the similarity measure
            should be computed.

    Returns:
        The affine measure if both the strings are not missing (i.e NaN or
        None), else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.affine('dva', 'deeva')
        1.5
        >>> em.affine(None, 'deeva')
        nan
    """
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    # Create the similarity measure object
    measure = sm.Affine()

    # if not isinstance(s1, six.string_types):
    #     s1 = six.u(str(s1))
    #
    # if isinstance(s1, bytes):
    #     s1 = s1.decode('utf-8', 'ignore')
    #
    # if not isinstance(s2, six.string_types):
    #     s2 = six.u(str(s2))
    #
    # if isinstance(s2, bytes):
    #     s2 = s2.decode('utf-8', 'ignore')

    s1 = gh.convert_to_str_unicode(s1)
    s2 = gh.convert_to_str_unicode(s2)

    # Call the function to compute the similarity
    return measure.get_raw_score(s1, s2)

def hamming_dist(s1, s2):
    """
    This function computes the Hamming distance between the two input
    strings.

    Args:
        s1,s2 (string): The input strings for which the similarity measure should
            be computed.

    Returns:
        The Hamming distance if both the strings are not missing (i.e NaN),
        else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.hamming_dist('alex', 'john')
        4
        >>> em.hamming_dist(None, 'john')
        nan


    """

    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    # Create the similarity measure object
    measure = sm.HammingDistance()

    s1 = gh.convert_to_str_unicode(s1)
    s2 = gh.convert_to_str_unicode(s2)


    # Call the function to compute the distance
    return measure.get_raw_score(s1, s2)


def hamming_sim(s1, s2):
    """
    This function computes the Hamming similarity between the two input
    strings.

    Args:
        s1,s2 (string): The input strings for which the similarity measure should
            be computed.

    Returns:
        The Hamming similarity if both the strings are not missing (i.e NaN),
        else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.hamming_sim('alex', 'alxe')
        0.5
        >>> em.hamming_sim(None, 'alex')
        nan

    """

    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    # Create the similarity measure object
    measure = sm.HammingDistance()

    s1 = gh.convert_to_str_unicode(s1)
    s2 = gh.convert_to_str_unicode(s2)

    # Call the function to compute the similarity score.
    return measure.get_sim_score(s1, s2)


def lev_dist(s1, s2):
    """
    This function computes the Levenshtein distance between the two input
    strings.

    Args:
        s1,s2 (string): The input strings for which the similarity measure should
            be computed.

    Returns:
        The Levenshtein distance if both the strings are not missing (i.e NaN),
        else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.lev_dist('alex', 'alxe')
        2
        >>> em.lev_dist(None, 'alex')
        nan

    """

    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    # Create the similarity measure object
    measure = sm.Levenshtein()

    s1 = gh.convert_to_str_unicode(s1)
    s2 = gh.convert_to_str_unicode(s2)

    # Call the function to compute the distance measure.
    return measure.get_raw_score(s1, s2)


def lev_sim(s1, s2):
    """
    This function computes the Levenshtein similarity between the two input
    strings.

    Args:
        s1,s2 (string): The input strings for which the similarity measure should
            be computed.

    Returns:
        The Levenshtein similarity if both the strings are not missing (i.e
        NaN), else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.lev_sim('alex', 'alxe')
        0.5
        >>> em.lev_dist(None, 'alex')
        nan

    """

    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    # Create the similarity measure object
    measure = sm.Levenshtein()

    s1 = gh.convert_to_str_unicode(s1)
    s2 = gh.convert_to_str_unicode(s2)

    # Call the function to compute the similarity measure
    return measure.get_sim_score(s1, s2)


def jaro(s1, s2):
    """
    This function computes the Jaro measure between the two input
    strings.

    Args:
        s1,s2 (string): The input strings for which the similarity measure should
            be computed.

    Returns:
        The Jaro measure if both the strings are not missing (i.e NaN),
        else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.jaro('MARTHA', 'MARHTA')
        0.9444444444444445
        >>> em.jaro(None, 'MARTHA')
        nan
    """

    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    # Create the similarity measure object
    measure = sm.Jaro()

    s1 = gh.convert_to_str_unicode(s1)
    s2 = gh.convert_to_str_unicode(s2)

    # Call the function to compute the similarity measure
    return measure.get_raw_score(s1, s2)


def jaro_winkler(s1, s2):
    """
    This function computes the Jaro Winkler measure between the two input
    strings.

    Args:
        s1,s2 (string): The input strings for which the similarity measure should
            be computed.

    Returns:
        The Jaro Winkler measure if both the strings are not missing (i.e NaN),
        else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.jaro_winkler('MARTHA', 'MARHTA')
        0.9611111111111111
        >>> >>> em.jaro_winkler('MARTHA', None)
        nan

    """

    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    # Create the similarity measure object
    measure = sm.JaroWinkler()

    s1 = gh.convert_to_str_unicode(s1)
    s2 = gh.convert_to_str_unicode(s2)

    # Call the function to compute the similarity measure
    return measure.get_raw_score(s1, s2)


def needleman_wunsch(s1, s2):
    """
    This function computes the Needleman-Wunsch measure between the two input
    strings.

    Args:
        s1,s2 (string): The input strings for which the similarity measure should
            be computed.

    Returns:
        The Needleman-Wunsch measure if both the strings are not missing (i.e
        NaN), else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.needleman_wunsch('dva', 'deeva')
        1.0
        >>> em.needleman_wunsch('dva', None)
        nan


    """

    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    # Create the similarity measure object
    measure = sm.NeedlemanWunsch()

    s1 = gh.convert_to_str_unicode(s1)
    s2 = gh.convert_to_str_unicode(s2)

    # Call the function to compute the similarity measure
    return measure.get_raw_score(s1, s2)


def smith_waterman(s1, s2):
    """
    This function computes the Smith-Waterman measure between the two input
    strings.

    Args:
        s1,s2 (string): The input strings for which the similarity measure should
            be computed.

    Returns:
        The Smith-Waterman measure if both the strings are not missing (i.e
        NaN), else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.smith_waterman('cat', 'hat')
        2.0
        >>> em.smith_waterman('cat', None)
        nan
    """

    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN

    # Create the similarity measure object
    measure = sm.SmithWaterman()

    s1 = gh.convert_to_str_unicode(s1)
    s2 = gh.convert_to_str_unicode(s2)

    # Call the function to compute the similarity measure
    return measure.get_raw_score(s1, s2)

def exact_match(s1, s2):
    """
    This function check if two objects are match exactly. Typically the
    objects are string, boolean and ints.

    Args:
        d1,d2 (str, boolean, int): The input objects which should checked
            whether they match exactly.

    Returns:
        A value of 1 is returned if they match exactly,
        else returns 0. Further if one of the objects is NaN or None,
        it returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.exact_match('Niall', 'Neal')
        0
        >>> em.exact_match('Niall', 'Niall')
        1
        >>> em.exact_match(10, 10)
        1
        >>> em.exact_match(10, 20)
        0
        >>> em.exact_match(True, True)
        1
        >>> em.exact_match(False, True)
        0
        >>> em.exact_match(10, None)
        nan
    """
    if s1 is None or s2 is None:
        return pd.np.NaN
    if pd.isnull(s1) or pd.isnull(s2):
        return pd.np.NaN
    # Check if they match exactly
    if s1 == s2:
        return 1
    else:
        return 0

# Token-based measures
def jaccard(arr1, arr2):
    """
    This function computes the Jaccard measure between the two input
    lists/sets.

    Args:
        arr1,arr2 (list or set): The input list or sets for which the Jaccard
            measure should be computed.

    Returns:
        The Jaccard measure if both the lists/set are not None and do not have
        any missing tokens (i.e NaN), else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.jaccard(['data', 'science'], ['data'])
        0.5
        >>> em.jaccard(['data', 'science'], None)
        nan
    """

    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN
    # Create jaccard measure object
    measure = sm.Jaccard()
    # Call a function to compute a similarity score
    return measure.get_raw_score(arr1, arr2)


def cosine(arr1, arr2):
    """
    This function computes the cosine measure between the two input
    lists/sets.

    Args:
        arr1,arr2 (list or set): The input list or sets for which the cosine
         measure should be computed.

    Returns:
        The cosine measure if both the lists/set are not None and do not have
        any missing tokens (i.e NaN), else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.cosine(['data', 'science'], ['data'])
        0.7071067811865475
        >>> em.cosine(['data', 'science'], None)
        nan

    """

    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN
    # Create cosine measure object
    measure = sm.Cosine()
    # Call the function to compute the cosine measure.
    return measure.get_raw_score(arr1, arr2)


def overlap_coeff(arr1, arr2):
    """
    This function computes the overlap coefficient between the two input
    lists/sets.

    Args:
        arr1,arr2 (list or set): The input lists or sets for which the overlap
            coefficient should be computed.

    Returns:
        The overlap coefficient if both the lists/sets are not None and do not
        have any missing tokens (i.e NaN), else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.overlap_coeff(['data', 'science'], ['data'])
        1.0
        >>> em.overlap_coeff(['data', 'science'], None)
        nan

    """

    #print "arr1:", arr1
    #print "arr2:", arr2
    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN
    # Create overlap coefficient measure object
    measure = sm.OverlapCoefficient()
    # Call the function to return the overlap coefficient
    return measure.get_raw_score(arr1, arr2)

def dice(arr1, arr2):
    """
    This function computes the Dice score between the two input
    lists/sets.

    Args:
        arr1,arr2 (list or set): The input list or sets for which the Dice
            score should be computed.

    Returns:
        The Dice score if both the lists/set are not None and do not
        have any missing tokens (i.e NaN), else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.dice(['data', 'science'], ['data'])
        0.6666666666666666
        >>> em.dice(['data', 'science'], None)
        nan

    """

    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN

    # Create Dice object
    measure = sm.Dice()
    # Call the function to return the dice score
    return measure.get_raw_score(arr1, arr2)

# Hybrid measure
def monge_elkan(arr1, arr2):
    """
    This function computes the Monge-Elkan measure between the two input
    lists/sets. Specifically, this function uses Jaro-Winkler measure as the
    secondary function to compute the similarity score.

    Args:
        arr1,arr2 (list or set): The input list or sets for which the
            Monge-Elkan measure should be computed.

    Returns:
        The Monge-Elkan measure if both the lists/set are not None and do not
        have any missing tokens (i.e NaN), else  returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.monge_elkan(['Niall'], ['Neal'])
        0.8049999999999999
        >>> em.monge_elkan(['Niall'], None)
        nan
    """

    if arr1 is None or arr2 is None:
        return pd.np.NaN
    if not isinstance(arr1, list):
        arr1 = [arr1]
    if any(pd.isnull(arr1)):
        return pd.np.NaN
    if not isinstance(arr2, list):
        arr2 = [arr2]
    if any(pd.isnull(arr2)):
        return pd.np.NaN
    # Create Monge-Elkan measure object
    measure = sm.MongeElkan()
    # Call the function to compute the Monge-Elkan measure
    return measure.get_raw_score(arr1, arr2)


def rel_diff(d1, d2):
    """
    This function computes the relative difference between two numbers

    Args:
        d1,d2 (float): The input numbers for which the relative difference
         must be computed.

    Returns:
        A float value of relative difference between the input numbers (if
        they are valid). Further if one of the input objects is NaN or None,
        it returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.rel_diff(100, 200)
        0.6666666666666666
        >>> em.rel_diff(100, 100)
        0.0
        >>> em.rel_diff(100, None)
        nan
    """

    if d1 is None or d2 is None:
        return pd.np.NaN
    if pd.isnull(d1) or pd.isnull(d2):
        return pd.np.NaN
    try:
        d1 = float(d1)
        d2 = float(d2)
    except ValueError:
        return pd.np.NaN
    if d1 == 0.0 and d2 == 0.0:
        return 0
    else:
        # Compute the relative difference between two numbers
        # ref: https://en.wikipedia.org/wiki/Relative_change_and_difference
        x = (2*abs(d1 - d2)) / (d1 + d2)
        return x


# compute absolute norm similarity
def abs_norm(d1, d2):
    """
    This function computes the absolute norm similarity between two numbers

    Args:
        d1,d2 (float): Input numbers for which the absolute norm must
            be computed.

    Returns:
        A float value of absolute norm between the input numbers (if
        they are valid). Further if one of the input objects is NaN or None,
        it returns NaN.

    Examples:
        >>> import py_entitymatching as em
        >>> em.abs_norm(100, 200)
        0.5
        >>> em.abs_norm(100, 100)
        1.0
        >>> em.abs_norm(100, None)
        nan

    """

    if d1 is None or d2 is None:
        return pd.np.NaN
    if pd.isnull(d1) or pd.isnull(d2):
        return pd.np.NaN
    try:
        d1 = float(d1)
        d2 = float(d2)
    except ValueError:
        return pd.np.NaN
    if d1 == 0.0 and d2 == 0.0:
        return 0
    else:
        # Compute absolute norm similarity between two numbers.
        x = (abs(d1 - d2) / max(d1, d2))
        if x <= 10e-5:
            x = 0
        return 1.0 - x
