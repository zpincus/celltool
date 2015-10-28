# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Tools for reading and writing delimited data files.

Files with various different delimiters and line endings can be read in, and
the presence of a header row can be automatically detected.
"""

import csv
import io
import types
import re
import numpy

def write_data_file(rows, filename, delimiter = ',', newline = '\n', generalize_floats = True):
    """Write data rows to a given file, separated by some delimiter.
    
    Parameters:
        - rows: a list of lists. Each inner list is a 'row' of data elements to be
            written to a single line of the file, separated by the delimiter.
        - delimiter: value with which to separate the data elements.
        - newline: value with which to separate the row.
        - generalize_floats: if true, format all floats with %g, to pretty-up the
            resulting file.
    """
    if generalize_floats:
        new_rows = []
        for row in rows:
            new_row = []
            for elem in row:
                if elem is None:
                    new_row.append('')
                else:
                    try:
                        new_row.append('%g'%elem)
                    except:
                        new_row.append(str(elem))
            new_rows.append(new_row)
        rows = new_rows
    f = open(filename, 'w')
    writer = csv.writer(f, delimiter = ',', lineterminator=newline)
    writer.writerows(rows)
    f.close()

class DataFile(object):
    """Class for reading delimited data files from disk."""
    
    def __init__(self, filename, delimiter = None, type_hierarchy = [int, float],
        skip_empty = True, type_dict = {}):
        """Read a data file as a delimited table of numbers and/or strings.
        
        If the delimiter parameter is provided, parse the file as rows of elements
        so delimited. If no parameter is provided, make an attempt to determine if 
        the delimiter is one of TAB, COMMA, COLON or SPACE. The input data can have
        UNIX, traditional-mac or windows-style line endings. Completely empty rows
        (with or without delimiters) can optionally be skipped.
        
        After the file is read, try to coerce each element to a numeric type.
        The type coercions attempted, and their order, are controlled by the
        type_hierarchy parameter. If all coercion attempts fail, the element is
        left as a string. If an element is the empty string, then it is 'coerced'
        to None.
        
        The coerced values are stored as a list-of-lists in the 'data' instance 
        variable, and their types are stored in the 'types' variable. These variables
        are lists of rows, where each row is a list of elements. Each row is guaranteed
        to be the same length; if some rows in the input data are too short, the rows
        are padded with None values (and NoneType types in the 'types' variable).
        
        If the type_dict parameter is provided, it should be a dictionary mapping
        column numbers (zero-indexed, please!) to types. For any rows so specified,
        the data values will be converted to that type and coercion will not be
        attempted.
        """
        self.type_hierarchy = type_hierarchy
        self.type_dict = type_dict
        f = open(filename, 'rU')
        self.lines = f.read()
        f.close()
        dialect = csv.Sniffer().sniff(self.lines, ['\t', ',', ';', ' '])
        if delimiter:
            reader = csv.reader(io.StringIO(self.lines), dialect, delimiter = delimiter)
        else:
            reader = csv.reader(io.StringIO(self.lines), dialect)
        self.data = []
        self.types = []
        for row in reader:
            if skip_empty and len(row) == 0 or numpy.alltrue([len(elem) == 0 for elem in row]):
                continue
            row_values, row_types = self._coerce(row)
            self.data.append(row_values)
            self.types.append(row_types)
        max_len = max([len(row) for row in self.data])
        for data_row, type_row in zip(self.data, self.types):
            l = len(data_row)
            if l < max_len:
                data_row.extend([None]*(max_len - l))
                type_row.extend([type(None)]*(max_len - l))
    
    def _coerce(self, row):
        """Coerce a list of strings to a list of numeric types, and return (coerced_values, coerced_types).
        
        This method uses the type_hierarchy instance variable to determine the
        specific type coercions attempted and their order. If an empty string is
        passed, it is 'coerced' to None, and NoneType is recorded in the coerced_types
        list. Similarly, if no coercion can be performed successfully, the string value
        and 'str' are recorded.
        
        This method uses the type_dict instance variable to determine if any columns
        exist whose desired types are already known, and skips coercion in those
        cases.
        """
        coerced_values = []
        coerced_types = []
        for i, elem in enumerate(row):
            if i in self.type_dict:
                the_type = self.type_dict[i]
                coerced_values.append(the_type(elem))
                coerced_types.append(the_type)
                continue
            elif elem == '':
                coerced_values.append(None)
                coerced_types.append(type(None))
                continue
            coerced = False
            for this_type in self.type_hierarchy:
                try:
                    coerced_values.append(this_type(elem))
                    coerced_types.append(this_type)
                    coerced = True
                    break
                except:
                    pass
            if not coerced:
                coerced_values.append(elem)
                coerced_types.append(str)
        return coerced_values, coerced_types
    
    def has_header(self):
        """Return whether the data is likely to include a header row.
        
        Whether there is a header row is determined heuristically as follows:
            (1) Determine the majority type for each column (this count excludes the first row)
            (2) If any the types of the elements in the first row are NOT
                    identical to the majority type for that column, then the first row is
                    assumed to be a header.
        """
        type_dicts = [{} for elem in self.data[0]]
        for type_row in self.types[1:]:
            for t, td in zip(type_row, type_dicts):
                try:
                    td[t] += 1
                except:
                    td[t] = 0
        majority_types = [max([(count, name) for name, count in type_dict.items()])[1] for type_dict in type_dicts]
        header_score = sum([t != h for t, h in zip(majority_types, self.types[0])])
        return header_score > 0
    
    def get_header_and_data(self):
        """Return a (header, data) tuple, where 'header' is None if the file does not have a header row.
        
        If the file has a header row (as determined by has_header, return (header, data)
        where 'data' consist of the remaining rows of the data attribute. Otherwise, 
        return (None, data) where 'data' is all of the data rows.
        
        If a header was detected, the rows returned as 'data' can either be accessed by
        index (e.g. value = data[0][0]), or by the name of the element in the header
        (e.g. value = data[0]['column name']).
        """
        if self.has_header():
            header = self.data[0]
            header_dict = _NamedRow.make_header_from_list(header)
            return header, [_NamedRow(header_dict, row) for row in self.data[1:]]
        else:
            return None, self.data
        
    def write(self, filename, delimiter = ',', newline = '\n'):
        """Write the data out to a file with the specified delimiter and newline characters."""
        write_data_file(self.data, filename, delimiter, newline)

class _NamedRow(list):
    def make_header_from_list(header_list):
        return dict([(name, i) for i, name in enumerate(header_list)])
    make_header_from_list = staticmethod(make_header_from_list)
    def __init__(self, header, row):
        self.header = header
        list.__init__(self, row)
    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except:
            return list.__getitem__(self, self.header[index])
