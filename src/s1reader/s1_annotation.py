import datetime
import xml.etree.ElementTree as ET
import os
import zipfile
import glob
import numpy as np



class annotation:
    # A parent class for annotation reader for Calibrarion, Noise, and poteltially L1
    def __init__(self,path_annotation:str,polarization='VH',subswath=1):
        #Attributes
        self._IPF_version=None
        self.path_annotation=path_annotation #Accepted format: SAFE .zip file, SAFE directory, or annotation .xml file
        self.polarization=polarization
        self.subswath=subswath #Subswath of the annotation file to load
        self._xml_loaded=None
        self.kind=''

        #Data access point for the annotation .xml file
        self.xml_et=None

        #TODO: Check the SAFE name and estimate the expected polarization. 
        #See: https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-product-formatting
    
    def _get_search_phrase(self):
        str_safe=os.path.basename(self.path_annotation).replace('.zip','')
        token_safe=str_safe.split('_')
        if len(token_safe)<3:
            raise ValueError(f'Invalid path_annotation: {self.path_annotation}')
        else:
            phrase_base='{PLATFORM}-{MODE}{SUBSWATH}-{TYPE}-{POL}'.format(PLATFORM=token_safe[0].lower(),
                    MODE=token_safe[1].lower(),
                    SUBSWATH=self.subswath,
                    TYPE=token_safe[2].lower(),
                    POL=self.polarization.lower())

            if self.kind=='calibration' or self.kind=='noise':
                phrase_xml_to_search=f'{self.kind}-{phrase_base}'
            elif self.kind=='':
                phrase_xml_to_search=f'/{phrase_base}'
            else:
                raise ValueError(f'Cannot recognize the annotation type {self.kind}')

            return phrase_xml_to_search


    def _load_xml_as_et(self): #kind: ['', 'calibration', 'noise']
        if os.path.isfile(self.path_annotation):
            if self.path_annotation.endswith('.zip'):
                #zip_in=zipfile.ZipFile(self.path_annotation,'r')
                with zipfile.ZipFile(self.path_annotation,'r') as zip_in:
                    phrase_xml_in_zip=self._get_search_phrase()

                    #search for the .xml file
                    filename_xml=None
                    for fileinfo in zip_in.filelist:
                        if (phrase_xml_in_zip in fileinfo.filename) and (fileinfo.filename.endswith('.xml')):
                            filename_xml=fileinfo.filename
                            break
                    
                    #load the xml if found.
                    if filename_xml is None:
                        raise ValueError('Cannot find the annotation file.')
                    else:
                        self.xml_et=ET.fromstring(zip_in.read(filename_xml))
                    
                    #finalize
                    self._xml_loaded=filename_xml
                #zip_in.close()

            elif self.path_annotation.endswith('.xml'):
                #load the xml as file
                self.xml_et=ET.parse(self.path_annotation)
                self._xml_loaded=self.xml_et
            else:
                raise ValueError(f'Invalid path for the annotation: {self.path_annotation}')

                

        elif os.path.isdir(self.path_annotation):
            phrase_xml_in_dir=self._get_search_phrase()
            if self.kind=='':
                list_file_xml=glob.glob(f'{self.path_annotation}/annotation/{phrase_xml_in_dir}*.xml')
            elif self.kind=='calibration' or self.kind=='noise':
                list_file_xml=glob.glob(f'{self.path_annotation}/annotation/calibration/{phrase_xml_in_dir}*.xml')
            else:
                raise ValueError(f'Annotation kind was not provided or not recognized: {self.kind}')

            if len(list_file_xml)==1:
                self.xml_et=ET.parse(list_file_xml[0])
                self._xml_loaded=list_file_xml[0]
                
            elif len(list_file_xml)==0:
                raise ValueError(f'Cannot find the corresponding .xml file')
            else:
                #raise ValueError(f'Search for the annotation file does not look right. len={len(list_file_xml)}')
                raise ValueError('More than one .xml files are found:\n'+'\n'.join(list_file_xml))


    def _parse_VectorList(self,name_VectorList:str,name_vector:str,str_type:str):
        #NOTE: str type: ['datetime','scalar_integer','scalar_float','vector_integer','vector_float','str']
        if self.xml_et is None:
            self._load_xml_as_et()
        
        element_to_parse=self.xml_et.find(name_VectorList)
        num_element=len(element_to_parse)

        list_out=[None]*num_element
        
        if str_type=='datetime':
            for i,elem in enumerate(element_to_parse):
                str_elem=elem.find(name_vector).text
                list_out[i]=datetime.datetime.strptime(str_elem,'%Y-%m-%dT%H:%M:%S.%f')
        
        elif str_type=='scalar_int':
            for i,elem in enumerate(element_to_parse):
                str_elem=elem.find(name_vector).text
                list_out[i]=int(str_elem)

        elif str_type=='scalar_float':
            for i,elem in enumerate(element_to_parse):
                str_elem=elem.find(name_vector).text
                list_out[i]=float(str_elem)

        elif str_type=='vector_int':
            for i,elem in enumerate(element_to_parse):
                str_elem=elem.find(name_vector).text
                list_out[i]=np.array([int(strin) for strin in str_elem.split()])

        elif str_type=='vector_float':
            for i,elem in enumerate(element_to_parse):
                str_elem=elem.find(name_vector).text
                list_out[i]=np.array([float(strin) for strin in str_elem.split()])
        
        elif str_type=='str':
            list_out=element_to_parse[0].find(name_vector).text
        
        else:
            raise ValueError(f'Cannot recognize the type of the element: {str_type}')

        return list_out


    @property
    def IPF_version(self):
        if self._IPF_version is None:
            if os.path.isfile(self.path_annotation):
                str_safe=os.path.basename(self.path_annotation).replace('.zip','')
                if self.path_annotation.endswith('.zip'):
                    zip_in=zipfile.ZipFile(self.path_annotation,'r')
                    str_manifest=zip_in.read(f'{str_safe}.SAFE/manifest.safe').decode()
            elif os.path.isdir(self.path_annotation):
                with open(f'{self.path_annotation}/manifest.safe','r') as manifest_in:
                    str_manifest=manifest_in.read()
            else:
                raise ValueError(f'Cannot find the manifest.safe file using path_annotation: {self.path_annotation}')

            lines_manifest=str_manifest.split('\n')
            for line_manifest in lines_manifest:
                if '<safe:software ' in line_manifest:
                    self._IPF_version=line_manifest.split('version=')[1].replace('"','').replace('/>','')
                    break
        
        return self._IPF_version



class calibration(annotation):
    def __init__(self,path_annotation,polarization='VH',subswath=1):
        super().__init__(path_annotation,polarization,subswath)
        self.kind='calibration'

        #on-the-fly attributes
        self._list_azimuthTime=None
        self._list_line=None
        self._list_pixel=None
        self._list_sigmaNought=None
        self._list_betaNought=None
        self._list_gamma=None
        self._list_dn=None
    
    @property
    def list_azimuthTime(self):
        if self._list_azimuthTime is None:
            self._list_azimuthTime=self._parse_VectorList('calibrationVectorList','azimuthTime','datetime')
        return self._list_azimuthTime


    @property
    def list_line(self):
        if self._list_line is None:
            self._list_line=self._parse_VectorList('calibrationVectorList','line','scalar_int')
        return self._list_line


    @property
    def list_pixel(self):
        if self._list_pixel is None:
            self._list_pixel=self._parse_VectorList('calibrationVectorList','pixel','vector_int')
        return self._list_pixel



    @property
    def list_sigmaNought(self):
        if self._list_sigmaNought is None:
            self._list_sigmaNought=self._parse_VectorList('calibrationVectorList','sigmaNought','vector_float')
        return self._list_sigmaNought

    
    @property
    def list_betaNought(self):
        if self._list_betaNought is None:
            self._list_betaNought=self._parse_VectorList('calibrationVectorList','betaNought','vector_float')
        return self._list_betaNought

    
    @property
    def list_gamma(self):
        if self._list_gamma is None:
            self._list_gamma=self._parse_VectorList('calibrationVectorList','gamma','vector_float')
        return self._list_gamma


    @property
    def list_dn(self):
        if self._list_dn is None:
            self._list_dn=self._parse_VectorList('calibrationVectorList','dn','vector_float')
        return self._list_dn


class noise(annotation):
    #TODO: Schema of the NADS is slightly different before/after IPF version 2.90. Needs to be adaptive in accordance with the version.
    #in ISCE2 code: if float(self.IPFversion) < 2.90:
    # REF: .../isce2/components/isceobj/Sensor/GRD/Sentinel1.py

    def __init__(self,path_annotation,polarization='VH',subswath=1):
        super().__init__(path_annotation,polarization,subswath)
        self.kind='noise'

        #on-the-fly attributes
        self._rg_list_azimuthTime=None
        self._rg_list_line=None
        self._rg_list_pixel=None
        self._rg_list_NoiseRangeLut=None
        self._az_swath=None
        self._az_firstAzimuthLine=None
        self._az_firstRangeSample=None
        self._az_lastAzimuthLine=None
        self._az_lastRangeSample=None
        self._az_line=None
        self._az_noiseAzimuthLut=None
    
    @property
    def rg_list_azimuthTime(self):
        if self._rg_list_azimuthTime is None:
            self._rg_list_azimuthTime=self._parse_VectorList('noiseRangeVectorList','azimuthTime','datetime')
        return self._rg_list_azimuthTime

    @property
    def rg_list_line(self):
        if self._rg_list_line is None:
            self._rg_list_line=self._parse_VectorList('noiseRangeVectorList','line','scalar_int')
        return self._rg_list_line

    @property
    def rg_list_pixel(self):
        if self._rg_list_pixel is None:
            self._rg_list_pixel=self._parse_VectorList('noiseRangeVectorList','pixel','vector_int')
        return self._rg_list_pixel

    @property
    def rg_list_NoiseRangeLut(self):
        if self._rg_list_NoiseRangeLut is None:
            self._rg_list_NoiseRangeLut=self._parse_VectorList('noiseRangeVectorList','noiseRangeLut','vector_float')
        return self._rg_list_NoiseRangeLut

    @property
    def az_swath(self):
        if self._az_swath is None:
            self._az_swath=self._parse_VectorList('noiseAzimuthVectorList','swath','str')
        return self._az_swath

    @property
    def az_firstAzimuthLine(self):
        if self._az_firstAzimuthLine is None:
            self._az_firstAzimuthLine=self._parse_VectorList('noiseAzimuthVectorList','firstAzimuthLine','scalar_int')[0]
        return self._az_firstAzimuthLine
    
    @property
    def az_firstRangeSample(self):
        if self._az_firstRangeSample is None:
            self._az_firstRangeSample=self._parse_VectorList('noiseAzimuthVectorList','firstRangeSample','scalar_int')[0]
        return self._az_firstRangeSample

    @property
    def az_lastAzimuthLine(self):
        if self._az_lastAzimuthLine is None:
            self._az_lastAzimuthLine=self._parse_VectorList('noiseAzimuthVectorList','lastAzimuthLine','scalar_int')[0]
        return self._az_lastAzimuthLine
    
    @property
    def az_lastRangeSample(self):
        if self._az_lastRangeSample is None:
            self._az_lastRangeSample=self._parse_VectorList('noiseAzimuthVectorList','lastRangeSample','scalar_int')[0]
        return self._az_lastRangeSample

    @property
    def az_line(self):
        if self._az_line is None:
            self._az_line=self._parse_VectorList('noiseAzimuthVectorList','line','vector_int')[0]
        return self._az_line

    @property
    def az_noiseAzimuthLut(self):
        if self._az_noiseAzimuthLut is None:
            self._az_noiseAzimuthLut=self._parse_VectorList('noiseAzimuthVectorList','noiseAzimuthLut','vector_float')
        return self._az_noiseAzimuthLut

