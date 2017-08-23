class TruthData(object):
    """  
        What follows is a dictionary of all the XML tags that
        are used in the Truth Data XML that is generated from
        the Radiologist annotated truth data in the LIDC data
        set.
    """

    ROOT_TAG = 'LidcTruthData'
    HEADER_TAG = 'TruthDataHeader'
    BODY_TAG = 'TruthData'
    
    # All header nodes follow
    HEADER_PATIENTID_TAG = 'PatientId'
    HEADER_UNBLINDED_NODULE_COUNT = 'UnblindedNoduleCount'
    HEADER_NONNODULE_COUNT = 'NonNoduleCount'
    HEADER_GENERATION_DATE = 'GenerationDate'
    HEADER_GENERATION_TIME = 'GenerationTime'
    HEADER_SOURCELIDC_XML = 'SourceLidcXml'
    HEADER_STUDYINSTANCE_ID = 'StudyInstanceUID'
    HEADER_SERIESINSTANCE_ID = 'SeriesInstanceUID'
    
    # All body nodes follow
    
    # Body Nodule Elements below
    BODY_UNBLINDEDREADNODULE = 'UnblindedReadNodule'
    BODY_NODULEID = 'NoduleId'
    BODY_ROI = 'Roi'
    BODY_ROI_SOPUID = 'ImageSopUid'
    BODY_ROI_ZPOS = 'ImageZPosition'
    BODY_ROI_INCLUSION = 'Inclusion'
    BODY_ROI_EDGEMAP = 'EdgeMap'
    BODY_ROI_EDGEMAP_X = 'X'
    BODY_ROI_EDGEMAP_Y = 'Y'
    
    # Body Non Nodule Elements below
    BODY_NONNODULE = 'NonNodule'
    BODY_NONNODULE_ID = 'NonNoduleId'
    BODY_NONNODULE_ZPOS = 'ImageZPosition'
    BODY_NONNODULE_SOPUID = 'ImageSopUid'
    BODY_NONNODULE_CENTROID = 'Centroid'
    BODY_NONNODULE_CENTROID_X = 'X'
    BODY_NONNODULE_CENTROID_Y = 'Y'
    
    patientId = ''
    unblindedNoduleCount = 0
    nonnoduleCount = 0
    generationDate = ''
    generationTime = ''
    sourceLidcXml = ''
    studyInstanceId = ''
    seriesInstanceId = ''
    
    def __init__(self, patientId, ubnoduleCount, nnCount, sourceXml, studyLidcInstanceId, seriesLidcInstanceId):
        self.patientId = patientId
        self.unblindedNoduleCount = ubnoduleCount
        self.nonnoduleCount = nnCount
        self.sourceLidcXml = sourceXml
        self.studyInstanceId = studyLidcInstanceId
        self.seriesInstanceId = seriesLidcInstanceId
    
    unblindedReadNodules = []
    nonNodules = []
    
    truthRecords = []
    
    def addTruthRecord(self, dcmImage, ubNodules, nonNodules):
        tr = TruthRecord(dcmImage, ubNodules, nonNodules)
        self.truthRecords.append(tr)
    
    def addUnblindedReadNodule(self, unblindedReadNodule):
        self.unblindedReadNodules.append(unblindedReadNodule)
        self.unblindedNoduleCount = len(self.unblindedReadNodules)
        
    def addNonNodule(self, nonNodule):
        self.nonNodules.append(nonNodule)
        self.nonnoduleCount = len(self.nonNodules)

class TruthRecord(object):
    def __init__(self, dcmImage, ubNodules, nonNodules):
        self.dcm_image = dcmImage
        self.unblinded_read_

# Coordinate for handling spatial truth data
class Point(object):
    pos_x = 0
    pos_y = 0   

# Represents an Unblinded Nodule annotation 
# in the DCM Radiologist annotated XMLs
class UnblindedReadNodule(object):
    noduleId = ''
    points = []
    pos_z = 0
    inclusion = True
    edgePoints = []
    image_sop_uid = ''
    
    def __init__(self, nodule_id, z_position, inclusion, image_sop_instance_id):
        self.noduleId = nodule_id
        self.pos_z = z_position
        self.inclusion = inclusion
        self.image_sop_uid = image_sop_instance_id
        
    def addEdgePoint(self, point):
        self.edgePoints.append(point)

# Represents an annotated Non-nodule in
# LIDC annotations.
class NonNodule(object):
    noduleId = ''
    pos_z = 0
    sop_image_uid = ''
    centroid_x = 0
    centroid_y = 0
    
    def __init__(self, nodule_id, z_position, centroid_x_position, centroid_y_position, sop_image_id):
        self.noduleId = nodule_id
        self.pos_z = z_position
        self.centroid_x = centroid_x_position
        self.centroid_y = centroid_y_position
        self.sop_image_uid = sop_image_id
        
# Represents a DCM image in the LIDC repository
class DCMImage(object):
    def __init__(self, dicomObj):
        self.dicom = dicomObj
        self.sopInstanceId = dicomObj.SOPInstanceUID
    
