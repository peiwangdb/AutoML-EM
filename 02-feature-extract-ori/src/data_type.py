import pandas as pd
import six

def get_type(column):
    """
     Given a pandas Series (i.e column in pandas DataFrame) obtain its type
    """
    # Validate input parameters
    # # We expect the input column to be of type pandas Series

    if not isinstance(column, pd.Series):
        raise AssertionError('Input (column) is not of type pandas series')

    # To get the type first drop all NaNa
    column = column.dropna()

    # Get type for each element and convert it into a set (and for
    # convenience convert the resulting set into a list)
    #print column.map(type).tolist()

    type_list = list(set(column.map(type).tolist()))
    type_count_dict = {}
    for t in type_list:
        if t not in type_count_dict:
            type_count_dict[t] = 0
        type_count_dict[t] += 1

    if len(type_count_dict) > 1:
        return "string"
    elif len(type_count_dict) == 1:
        if str in type_count_dict:
            return "string"
        elif bool in type_count_dict:
            return "boolean"
        else: 
            return "numeric"


    # If the list is empty, then we cannot decide anything about the column.
    # We will raise a warning and return the type to be numeric.
    # Note: The reason numeric is returned instead of a special type because,
    #  we want to keep the types minimal. Further, explicitly recommend the
    # user to update the returned types later.
    if len(type_list) == 0:
        logger.warning("Column {0} does not seem to qualify as any atomic type. "
                       "It may contain all NaNs. Please update the values of column {0}".format(column.name))
        return 'un_determined'

    # If the column qualifies to be of more than one type (for instance,
    # in a numeric column, some values may be inferred as strings), then we
    # will raise an error for the user to fix this case.
    if len(type_list) > 1:
        logger.warning('Column %s qualifies to be more than one type. \n'
                       'Please explicitly set the column type like this:\n'
                       'A["address"] = A["address"].astype(str) \n'
                       'Similarly use int, float, boolean types.' % column.name)
        raise AssertionError('Column %s qualifies to be more than one type. \n'
                             'Please explicitly set the column type like this:\n'
                             'A["address"] = A["address"].astype(str) \n'
                             'Similarly use int, float, boolean types.' % column.name)
    else:
        # the number of types is 1.
        returned_type = type_list[0]
        # Check if the type is boolean, if so return boolean
        if returned_type == bool or returned_type == pd.np.bool_:
            return 'boolean'

        # Check if the type is string, if so identify the subtype under it.
        # We use average token length to identify the subtypes

        # Consider string and unicode as same
        elif returned_type == str or returned_type == six.unichr or returned_type == six.text_type:
            return "string"
        else:
            # Finally, return numeric if it does not qualify for any of the
            # types above.
            return "numeric"