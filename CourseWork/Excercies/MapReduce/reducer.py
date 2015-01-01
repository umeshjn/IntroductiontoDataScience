import sys
import logging

from util import reducer_logfile
logging.basicConfig(filename=reducer_logfile, format='%(message)s',
                    level=logging.INFO, filemode='w')

def reducer():
    
    #Also make sure to fill out the mapper code before clicking "Test Run" or "Submit".

    #Each line will be a key-value pair separated by a tab character.
    #Print out each key once, along with the total number of Aadhaar 
    #generated, separated by a tab. Make sure each key-value pair is 
    #formatted correctly! Here's a sample final key-value pair: 'Gujarat\t5.0'

    #Since you are printing the output of your program, printing a debug 
    #statement will interfere with the operation of the grader. Instead, 
    #use the logging module, which we've configured to log to a file printed 
    #when you click "Test Run". For example:
    #logging.info("My debugging message")
        
    aadhaar_generated = 0
    oldkey = None 
    
    for line in sys.stdin:
        data = line.strip().split("\t")
        #logging.info(data)
        if len(data)!=2:
            continue
        thiskey, generated = data
        if oldkey and oldkey != thiskey:
            print "{0}\t{1}".format(oldkey,aadhaar_generated)
            aadhaar_generated =0
        oldkey = thiskey
        aadhaar_generated += float(generated)
        
    if oldkey != None:
            print "{0}\t{1}".format(oldkey,aadhaar_generated)

reducer()
