import numpy

def write_labeldata(labeldata, outputfile):
  header = numpy.array([0x0801, len(labeldata)], dtype='>i4')
  with open(outputfile, "wb") as f:
    f.write(header.tobytes())
    f.write(labeldata.tobytes())

def write_imagedata(imagedata, outputfile):
  header = numpy.array([0x0803, len(imagedata), 28, 28], dtype='>i4')
  with open(outputfile, "wb") as f:
    f.write(header.tobytes())
    f.write(imagedata.tobytes())