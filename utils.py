#!/usr/bin/python3.6

"""
This module provides helper functions.
"""

import sys
import re
from tqdm import tqdm
import time

def loop_input(rtype=str, default=None, msg=""):
    """
    Wrapper function for command-line input that specifies an input type
    and a default value.
    :param rtype: type of the input. one of str, int, float, bool
    :type rtype: type
    :param default: value to be returned if the input is empty
    :param msg: message that is printed as prompt
    :type msg: str
    :return: value of the specified type
    """
    while True:
        try:
            s = input(msg)
            if rtype == bool and len(s) > 0:
                if s=="True":
                    return True
                elif s=="False":
                    return False
                else:
                    print("Input needs to be convertable to",rtype,"-- try again.")
                    continue
            else:
                return rtype(s) if len(s) > 0 else default
        except ValueError:
            print("Input needs to be convertable to",rtype,"-- try again.")
            continue


class ConfigReader():
    """
    Basic container and management of parameter configurations.
    Read a config file (typically ending with .cfg), use this as container for
    the parameters during runtime, and change/write parameters.

    CONFIGURATION FILE SYNTAX
    - one parameter per line, containing a name and a value
    - name and value are separated by at least one white space or tab
    - names can contain alphanumeric symbols and underscores (no '-', please!)
    - mutiple space-separated values in one line are processed as a list
    - multiword expressions are marked with double quotation marks
    - lines starting with '#' are ignored
    - no in-line comments!
    - config files should have the extension 'cfg' (to indicate their purpose)
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.params = self.read_config()

    def __repr__(self):
        """
        returns tab-separated key-value pairs (one pair per line)
        """
        return "\n".join([str(k)+"\t"+str(v) for k,v in self.params.items()])

    def read_config(self):
        #TODO docstring
        cfg = {}
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        c = 0 #CLEANUP
        for line in lines:
            print(f"line {c}: {line}") #CLEANUP
            c += 1 #CLEANUP
            words = line.rstrip().split()
            if not words: # ignore empty lines
                continue
            elif words[0].startswith("#"): # ignore commented-out lines
                continue
            else:
                """ parse the line to get parameter name and parameter value """
                paramname = words.pop(0)
                if not words: # no value specified
                    print(f"WARNING: no value specified for parameter {paramname}.")
                    paramvalue = None
                elif words[0].startswith("["): # detects a list of values
                    paramvalue = self.listparse(" ".join(words))
                elif words[0].startswith('"'): # detects a multi-word string
                    paramvalue = self.stringparse(words)
                else:
                    """ only single values are valid! """
                    if len(words) > 1:
                        #TODO make this proper error handling (= throw an error)
                        print(f"ERROR while parsing {self.filepath} --",
                              f"too many values in line '{line}'.")
                        sys.exit()
                    else:
                        """ parse the single value """
                        paramvalue = self.numberparse(words[0]) # 'words' is still a list
                        paramvalue = self.boolparse(paramvalue)

                cfg[paramname] = paramvalue # adds the parameter to the config

        self.config = cfg
        return self.config

    @classmethod
    def listparse(cls, liststring):
        """
        #TODO docstring
        """
        tmp_list = []
        result = []

        # pre-processing
        for w in words:
            if w.startswith('['): w = w.lstrip('[')
            if w.endswith(']'):   w = w.rstrip(']')
            if w.endswith(','):   w = w.rstrip(',')
            tmp_list.append(w)

        tmp_str = "" # to handle multi-word strings within lists
        for e in tmp_list:
            # in case of quoted strings
            if e.startswith('"') or e.startswith('\''):
                e = e.lstrip('"').lstrip('\'')
                # deal with single-word quoted strings
                if e.endswith('"') or e.endswith('\''):
                    e = e.rstrip('"').rstrip('\'')
                    result.append(e)
                else:
                    tmp_str += e
            # in case of multi-word strings
            elif e.endswith('"') or e.endswith('\''):
                tmp_str += " "+e.rstrip('"').rstrip('\'')
                result.append(tmp_str)
                tmp_str = ""
            # for string words within multi-word strings
            elif tmp_str:
                tmp_str += " "+e

            else:
                e = cls.numberparse(e) # convert to number if possible
                e = cls.boolparse(e) # convert to bool if possible
                result.append(e)

        return result

    @staticmethod
    def stringparse(words):
        words[0] = words[0][1:] # delete opening quotation marks
        words[-1] = words[-1][:-1] #delete closing quotation marks
        return " ".join(words)

    @staticmethod
    def numberparse(string):
        """
        Tries to convert 'string' to a float or even int.
        Returns int/float if successful, or else the input string.
        """
        try:
            floaty = float(string)
            if int(floaty) == floaty:
                return int(floaty)
            else:
                return floaty
        except ValueError:
            return string

    @staticmethod
    def boolparse(string):
        if string == "True" or string == "False":
            return bool(string)
        else:
            return string

    def get_config(self):
        """
        returns the config as a dictionary
        """
        return self.params

    def get_params(self):
        """
        returns a list of parameter names
        """
        return [key for key in self.params.keys()]

    def get(self, *paramname):
        """
        returns a specific value or a tuple of values corresponding to the
        provided parameter name(s).
        """
        values = [self.params[name] for name in paramname]
        if len(values) == 1:
            return values[0]
        else:
            return tuple(values)

    def set(self, paramname, value):
        self.params.update({paramname:value})