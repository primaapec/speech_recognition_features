
import h5py
from shutil import rmtree
import numpy as np
import sidekit as sk


class SIDEKITFeaturesExtractor:

    def __init__(self, input_filepath_structure=None, feature_filename_structure=None, mel_feature_filename_structure=None, sampling_frequency=16000,
                 lower_frequency=200, higher_frequency_mel=3800, filter_bank="log", filter_bank_size=24, window_size=0.025,
                 shift=0.01, ceps_number=20, snr=40, pre_emphasis=0.97):

        self.__sk_lin_feature_extractor = sk.FeaturesExtractor(
            audio_filename_structure=input_filepath_structure,
            feature_filename_structure=feature_filename_structure,
            lower_frequency=lower_frequency,
            higher_frequency=sampling_frequency/2,
            filter_bank="lin",
            filter_bank_size=filter_bank_size,
            window_size=window_size,
            shift=shift,
            ceps_number=ceps_number,
            vad="snr",
            snr=snr,
            save_param=["cep", "fb"],
            sampling_frequency=sampling_frequency,
            keep_all_features=True,
            pre_emphasis=pre_emphasis,
        )
        self.__sk_mel_feature_extractor = sk.FeaturesExtractor(
            audio_filename_structure=input_filepath_structure,
            feature_filename_structure=mel_feature_filename_structure,
            lower_frequency=lower_frequency,
            higher_frequency=higher_frequency_mel,
            filter_bank="log",
            filter_bank_size=filter_bank_size,
            window_size=window_size,
            shift=shift,
            ceps_number=ceps_number,
            vad="snr",
            snr=snr,
            save_param=["vad", "energy", "cep", "fb"],
            sampling_frequency=sampling_frequency,
            keep_all_features=True,
            pre_emphasis=pre_emphasis,
        )
        self.input_filepath_structure = input_filepath_structure
        self.feature_filename_structure = feature_filename_structure
        self.mel_feature_filename_structure = mel_feature_filename_structure
        self.sampling_frequency = sampling_frequency
        self.lower_frequency = lower_frequency
        self.higher_frequency_mel = higher_frequency_mel
        self.filter_bank_size = filter_bank_size
        self.window_size = window_size
        self.shift = shift
        self.ceps_number = ceps_number
        self.snr = snr
        self.pre_emphasis = pre_emphasis

    @staticmethod
    def __map_key(key, filterbank):
        mapper = {
            'log':
            {
                'cep': 'MFCC',
                'cep_header': 'log_cep_header',
                'cep_mean': 'log_cep_mean',
                'cep_min_range': 'log_cep_min_range',
                'cep_std': 'log_cep_std',
                'energy': 'energy',
                'energy_header': 'energy_header',
                'energy_mean': 'energy_mean',
                'energy_min_range': 'energy_min_range',
                'energy_std': 'energy_std',
                'fb': 'log_fb',
                'fb_header': 'log_fb_header',
                'fb_mean': 'log_fb_mean',
                'fb_min_range': 'log_fb_min_range',
                'fb_std': 'log_fb_std',
                'vad': 'vad',
            },
            'lin':
            {
                'cep': 'LFCC',
                'cep_header': 'lin_cep_header',
                'cep_mean': 'lin_cep_mean',
                'cep_min_range': 'lin_cep_min_range',
                'cep_std': 'lin_cep_std',
                'energy': 'energy',
                'energy_header': 'energy_header',
                'energy_mean': 'energy_mean',
                'energy_min_range': 'energy_min_range',
                'energy_std': 'energy_std',
                'fb': 'lin_fb',
                'fb_header': 'lin_fb_header',
                'fb_mean': 'lin_fb_mean',
                'fb_min_range': 'lin_fb_min_range',
                'fb_std': 'lin_fb_std',
                'vad': 'vad'
            }
        }
        return mapper[filterbank][key]

    def extract_features(self, filenames):
        self.__sk_lin_feature_extractor.save_list(
            filenames, [0]*len(filenames))
        self.__sk_mel_feature_extractor.save_list(
            filenames, [0]*len(filenames))
        self.features = dict()
        for file in filenames:
            with h5py.File(self.feature_filename_structure.format(file)) as f:
                self.features[file] = {
                    self.__map_key(key, 'lin'): f[file+'/' + key][()] for key in f[file].keys()}
            with h5py.File(self.mel_feature_filename_structure.format(file)) as f:
                self.features[file].update({
                    self.__map_key(key, 'log'): f[file+'/' + key][()] for key in f[file].keys()})
        return self.features


if __name__ == "__main__":
    import os
    print(os.path.curdir)
    input_filepath_structure = os.path.curdir + '/input_audio/{}.wav'
    feature_filename_structure = os.path.curdir + '/test_output_features/{}.h5'
    mel_feature_filename_structure = os.path.curdir + '/test_mel_output_features/{}.h5'
    # print(os.path.realpath(os.path.curdir + '/test_output_features/'))
    if os.path.isdir(os.path.realpath(os.path.curdir + '/test_output_features')):
        rmtree(os.path.realpath(os.path.curdir + '/test_output_features'))
    if os.path.isdir(os.path.realpath(os.path.curdir + '/test_mel_output_features')):
        rmtree(os.path.realpath(os.path.curdir + '/test_mel_output_features'))
    extractor = SIDEKITFeaturesExtractor(
        input_filepath_structure, feature_filename_structure, mel_feature_filename_structure)
    print(extractor.extract_features(["hello", "believer_part"]))
    # extractor = sk.features_extractor.FeaturesExtractor('../input_audio/{}.wav',
    #                                                     '../output_features/{}.h5',
    #                                                     sampling_frequency=None,
    #                                                     lower_frequency=200,
    #                                                     higher_frequency=3800,
    #                                                     filter_bank="log",
    #                                                     filter_bank_size=24,
    #                                                     window_size=0.025,
    #                                                     shift=0.01,
    #                                                     ceps_number=20,
    #                                                     vad="snr",
    #                                                     snr=40,
    #                                                     pre_emphasis=0.97,
    #                                                     save_param=[
    #                                                         "vad", "energy", "cep", "fb"],
    #                                                     keep_all_features=True)

    # show_list = ["hello", "believer_part"]
    # channel_list = [0, 0]

    # extractor.save_list(show_list=show_list,
    #                     channel_list=channel_list)  # ,num_thread=4)
