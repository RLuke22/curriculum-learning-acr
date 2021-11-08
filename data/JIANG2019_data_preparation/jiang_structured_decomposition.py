### NOTE: This is a modification to the pumpp/task/chord.py file so that pumpp.task.ChordTransformer generates the
### structured components proposed in Jiang, 2019

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Chord recognition task transformer'''

import re
from itertools import product

import numpy as np
import mir_eval
import jams

from librosa import time_to_frames
from librosa.sequence import transition_loop

from .base import BaseTaskTransformer
from ..exceptions import ParameterError
from ..labels import LabelBinarizer, LabelEncoder, MultiLabelBinarizer

__all__ = ['ChordTransformer', 'SimpleChordTransformer', 'ChordTagTransformer']

pattern = re.compile(r'^([A-G][#b]?|N|X)(:(\w*))?(\(.*\))?(/([#b]*[0-9]*))?$')
degree_pattern = re.compile(r'^([#b]*)([0-9]*)$')
asterisk_pattern = re.compile(r'(\*([#b]*)([0-9]),?)+')
seventh_pattern = re.compile(r'(\*)?([#b]*7)')
ninth_pattern = re.compile(r'([#b]*)[29]')
eleventh_pattern = re.compile(r'([#b]*)(11|4)')
thirteenth_pattern = re.compile(r'([#b]*)(13|6)')
pitch_dict = {'B#': 0,'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4, 'E#': 5, 'F': 5,
            'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11}
degree_dict = {v: k for k, v in pitch_dict.items()}
degree_to_interval = {'1': 0, '2': 2, '3': 4, '4': 5, '5': 7, '6': 9, '7': 11, '8': 12, '9': 14, '11': 17}
modifier_to_interval = {'': 0, '#': 1, 'b': -1, 'bb': -2, '##': 2}
root_index = {'N': 0, 'B#': 1, 'C': 1, 'C#': 2, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 4, 'E': 5, 'Fb': 5, 'E#': 6, 'F': 6,
            'F#': 7, 'Gb': 7, 'G': 8, 'G#': 9, 'Ab': 9, 'A': 10, 'A#': 11, 'Bb': 11, 'B': 12, 'Cb': 12}
triad_index = {'N': 0, 'maj': 1, 'min': 2, 'sus4': 3, 'sus2': 4, 'dim': 5, 'aug': 6}
bass_index = root_index
seventh_index = {'N': 0, '7': 1, 'b7': 2, 'bb7': 3}
ninth_index = {'N': 0, '9': 1, '#9': 2, 'b9': 3}
eleventh_index = {'N': 0, '11': 1, '#11': 2}
thirteenth_index = {'N': 0, '13': 1, 'b13': 2}

# 301 chord type vocabulary dictionaries
root_index_301 = {'B#': 0, 'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4, 'E#': 5, 'F': 5,
            'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11}
quality_index_301 = {'maj': 0,
                     'min': 1,
                     'aug': 2,
                     'dim': 3,
                     'maj/3': 4,
                     'maj/5': 5,
                     'min/b3': 6,
                     'min/5': 7,
                     'maj7': 8,
                     '7': 9,
                     'min7': 10,
                     'dim7': 11,
                     'hdim7': 12,
                     'maj9': 13,
                     '9': 14,
                     'min9': 15,
                     '11': 16,
                     '13': 17,
                     'sus4': 18,
                     'sus2': 19,
                     'sus4(b7)': 20,
                     'maj/2': 21,
                     'maj/b7': 22,
                     'min/2': 23,
                     'min/b7': 24}

quality_index_337 = {'maj': 0,
                     'min': 1,
                     'aug': 2,
                     'dim': 3,
                     'maj/3': 4,
                     'maj/5': 5,
                     'min/b3': 6,
                     'min/5': 7,
                     'maj7': 8,
                     '7': 9,
                     'min7': 10,
                     'dim7': 11,
                     'hdim7': 12,
                     'maj9': 13,
                     '9': 14,
                     'min9': 15,
                     '11': 16,
                     '13': 17,
                     'sus4': 18,
                     'sus2': 19,
                     'sus4(b7)': 20,
                     'maj/2': 21,
                     'maj/b7': 22,
                     'min/2': 23,
                     'min/b7': 24,
                     'min6': 25,
                     'maj6': 26,
                     'minmaj7': 27}

def _pad_nochord(target, axis=-1):
    '''Pad a chord annotation with no-chord flags.

    Parameters
    ----------
    target : np.ndarray
        the input data

    axis : int
        the axis along which to pad

    Returns
    -------
    target_pad
        `target` expanded by 1 along the specified `axis`.
        The expanded dimension will be 0 when `target` is non-zero
        before padding, and 1 otherwise.
    '''
    ncmask = ~np.max(target, axis=axis, keepdims=True)

    return np.concatenate([target, ncmask], axis=axis)


class ChordTransformer(BaseTaskTransformer):
    '''Chord annotation transformers.

    This transformer uses a (pitch, root, bass) decomposition of
    chord annotations.

    Attributes
    ----------
    name : str
        The name of the chord transformer

    sr : number > 0
        The sampling rate of audio

    hop_length : int > 0
        The number of samples between each annotation frame

    sparse : bool
        If True, root and bass values are sparsely encoded as integers in [0, 12].
        If False, root and bass values are densely encoded as 13-dimensional booleans.

    See Also
    --------
    SimpleChordTransformer
    '''
    def __init__(self, name='chord', sr=22050, hop_length=512, sparse=False):
        '''Initialize a chord task transformer'''

        super(ChordTransformer, self).__init__(name=name,
                                               namespace='chord',
                                               sr=sr, hop_length=hop_length)

        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([list(range(12))])
        self._classes = set(self.encoder.classes_)
        self.sparse = sparse

        if self.sparse:
            self.register('root_triad', [None, 1], np.int)
            self.register('bass', [None, 1], np.int)
            self.register('seventh', [None, 1], np.int)
            self.register('ninth', [None, 1], np.int)
            self.register('eleventh', [None, 1], np.int)
            self.register('thirteenth', [None, 1], np.int)

    def empty(self, duration):
        '''Empty chord annotations

        Parameters
        ----------
        duration : number
            The length (in seconds) of the empty annotation

        Returns
        -------
        ann : jams.Annotation
            A chord annotation consisting of a single `no-chord` observation.
        '''
        ann = super(ChordTransformer, self).empty(duration)

        ann.append(time=0,
                   duration=duration,
                   value='N', confidence=0)

        return ann

    # Chord decomposition according to ISMIR 2019, Jiang et al.
    # All chords have a valid decomposition, with exception to "incomplete triads", which 
    # are mapped to a set of "no mapping" labels (-1)
    def decompose_chord(self, chord):
        root = 'N'
        triad = 'N'
        bass = 'N'
        seventh = 'N'
        ninth = 'N'
        eleventh = 'N'
        thirteenth = 'N'

        #Root, shorthand, ilist, and bass degree, respectively
        #match.group(1), match.group(3), match.group(4), match.group(6)
        match = pattern.match(chord)
        
        # Extract the root note
        root = match.group(1)
        # All zeros is the chord decomposition for the 'N' symbol
        if root == 'N':
            return 0,0,0,0,0,0,300,336
        # We consider the 'X' chord to be an "incomplete" triad
        elif root == 'X':
            return -1,-1,-1,-1,-1,-1,-1,-1
        
        # Extract the bass note
        if match.group(6):
            # If slash included, then determine bass note
            degree_match = degree_pattern.match(match.group(6))
            interval = modifier_to_interval[degree_match.group(1)] + degree_to_interval[degree_match.group(2)]
            bass = degree_dict[(pitch_dict[root] + interval) % 12]
        else:
            # Otherwise, bass is root note
            bass = root

        # Extract the triad
        # First case: Shorthand triad exists
        if match.group(3):
            if match.group(3) in ['dim', 'dim7', 'hdim7']:
                triad = 'dim'
            elif match.group(3) in ['min', 'min7', 'min6', 'min9', 'minmaj7', 'min11', 'min13']:
                triad = 'min'
            elif match.group(3) in ['aug', 'sus2', 'sus4']:
                triad = match.group(3)
            # Note that '5' is missing a 3rd, and hence an incomplete triad
            # Note that '1' is only the root, and hence an incomplete triad
            elif match.group(3) in ['1', '5']:
                return -1,-1,-1,-1,-1,-1,-1,-1
            elif match.group(3) in ['maj', '7', '9', 'maj7', 'maj9', '11', 'maj6', '13', 'maj11', 'maj13']:
                triad = 'maj'
            else:
                print("Error: ", chord)
                exit(0)
        # Second case: No shorthand, no interval list -- this is by default a major triad
        elif (match.group(3) and match.group(4)) is None:
            triad = 'maj'
        # Third case, No shorthand, but interval list exists. In this case, we consider it an "incomplete triad"
        else:
            return -1,-1,-1,-1,-1,-1,-1,-1
        
        # Determine if triad has to be N'ed out
        if match.group(4):
            asterisk_match = asterisk_pattern.search(match.group(4))
            if asterisk_match:
                # Get list of all asterisk'ed degrees
                asterisk_degrees = [st[1:] for st in asterisk_match.group(0).split(',')]
                is_incomplete = False
                for asterisk_degree in asterisk_degrees:
                    # degrees found in a triad
                    if asterisk_degree in ['1', '3', 'b3', '5']:
                        is_incomplete = True 
                        break 
                if is_incomplete:
                    return -1,-1,-1,-1,-1,-1,-1,-1
        
        # Determine the 7th note
        if match.group(3) in ['7', '9', 'min7', 'min9', 'hdim7', 'min13', 'min11', '11', '13']:
            seventh = 'b7'
        elif match.group(3) in ['maj7', 'maj9', 'minmaj7', 'maj11', 'maj13']:
            seventh = '7'
        elif match.group(3) == 'dim7':
            seventh = 'bb7'
        
        # Add any 7's from the interval list
        if match.group(4):
            seven_match = seventh_pattern.search(match.group(4))
            if seven_match:
                # Matches if there is an asterisk in the 7th interval. Note that the only "extended degrees"
                # which have asterisks in the MARL dataset is the 7th interval. (Don't have to do these checks for 9th/11th/13th/etc...)
                if seven_match.group(1):
                    seventh = 'N'
                else:
                    seventh = seven_match.group(2)
        # Add any sevenths from the bass list
        if match.group(6):
            coerce_seventh_pattern_match = '(' + match.group(6) + ')'
            seven_match = seventh_pattern.search(coerce_seventh_pattern_match)
            if seven_match and seventh == 'N':
                seventh = seven_match.group(2)

        # Determine the 9th note
        if match.group(3) in ['9', 'maj9', 'min9', 'min11', 'min13', '11', '13', 'maj11', 'maj13']:
            ninth = '9'
        # Add any 9's from the interval list
        if match.group(4):
            ninth_match = ninth_pattern.search(match.group(4))
            if ninth_match:
                ninth = ninth_match.group(1) + '9'
        # Add any ninths from the bass list
        if match.group(6):
            coerce_ninth_pattern_match = '(' + match.group(6) + ')'
            ninth_match = ninth_pattern.search(coerce_ninth_pattern_match)
            if ninth_match and ninth == 'N':
                ninth = ninth_match.group(1) + '9'

        # Determine the 11th note
        if match.group(3) in ['min11', 'min13', '11', '13', 'maj11', 'maj13']:
            eleventh = '11'
        # Add any 11's from the interval list
        if match.group(4):
            eleventh_match = eleventh_pattern.search(match.group(4))
            if eleventh_match:
                eleventh = eleventh_match.group(1) + '11'  
        # Add any elevenths from the bass list
        if match.group(6):
            coerce_eleventh_pattern_match = '(' + match.group(6) + ')'
            eleventh_match = eleventh_pattern.search(coerce_eleventh_pattern_match)
            if eleventh_match and eleventh == 'N':
                if eleventh_match.group(1) != 'b':
                    eleventh = eleventh_match.group(1) + '11'       

        # Determine the 13th note
        if match.group(3) in ['min6', 'maj6', 'min13', 'maj13', '13']:
            thirteenth = '13'
        # Add any 13's from the interval list
        if match.group(4):
            thirteenth_match = thirteenth_pattern.search(match.group(4))
            if thirteenth_match:
                thirteenth = thirteenth_match.group(1) + '13'  
        # Add any thirteenths from the bass list
        if match.group(6):
            coerce_thirteenth_pattern_match = '(' + match.group(6) + ')'
            thirteenth_match = thirteenth_pattern.search(coerce_thirteenth_pattern_match)
            if thirteenth_match and thirteenth == 'N':
                if thirteenth_match.group(1) not in ['#', 'bb']:
                    thirteenth = thirteenth_match.group(1) + '13' 

        # Finally return the indices
        root_triad_index = 0
        if triad_index[triad] != 0 and root_index[root] != 0:
            root_triad_index = (triad_index[triad] - 1)*12 + (root_index[root] - 1) + 1

        #From here, we will create the chord vocabulary of 301 and 337 chord types
        # First, knock out all the major variations
        if match.group(3) == 'maj' or match.group(3) == '' or match.group(3) is None:
            if match.group(6) == '3':
                quality_301 = 'maj/3'
                quality_337 = 'maj/3'
            elif match.group(6) == '5':
                quality_301 = 'maj/5'
                quality_337 = 'maj/5'
            elif match.group(6) == '2':
                quality_301 = 'maj/2'
                quality_337 = 'maj/2'
            elif match.group(6) == 'b7':
                quality_301 = 'maj/b7'
                quality_337 = 'maj/b7'
            else:
                quality_301 = 'maj'
                quality_337 = 'maj'
        elif match.group(3) == 'min':
            if match.group(6) == 'b3':
                quality_301 = 'min/b3'
                quality_337 = 'min/b3'
            elif match.group(6) == '5':
                quality_301 = 'min/5'
                quality_337 = 'min/5'
            elif match.group(6) == '2':
                quality_301 = 'min/2'
                quality_337 = 'min/2'
            elif match.group(6) == 'b7':
                quality_301 = 'min/b7'
                quality_337 = 'min/b7'
            else:
                quality_301 = 'min'
                quality_337 = 'min'
        elif match.group(3) == 'sus4':
            if match.group(4):
                sus4_degree_list = match.group(4)[1:-1]
                sus4_degrees = [st for st in sus4_degree_list.split(',')]
                is_b7 = False
                for sus4_degree in sus4_degrees:
                    # degrees found in a triad
                    if sus4_degree == 'b7':
                        is_b7 = True 
                        break 
                if is_b7:
                    quality_301 = 'sus4(b7)'
                    quality_337 = 'sus4(b7)'
                else:
                    quality_301 = 'sus4'
                    quality_337 = 'sus4'
            else:
                quality_301 = 'sus4'
                quality_337 = 'sus4'
            if quality_301 =='sus4' and match.group(6) == 'b7':
                quality_301 = 'sus4(b7)'
                quality_337 = 'sus4(b7)'
                
        elif match.group(3) in ['aug', 'dim', 'sus2', 'maj7', '7', 'min7', 'dim7', 'hdim7', 'maj9', '9', 'min9', '11', '13']:
            quality_301 = match.group(3)
            quality_337 = match.group(3)
        else:
            # OOD chords maps to all -1's
            quality_301 = 'OOD'
            quality_337 = 'OOD'
        
        if match.group(3) in ['min6', 'maj6', 'minmaj7']:
            quality_337 = match.group(3)

        vocab_301_index = -1
        if quality_301 != 'OOD':
            vocab_301_index = 25 * root_index_301[root] + quality_index_301[quality_301]

        vocab_337_index = -1
        if quality_337 != 'OOD':
            vocab_337_index = 28 * root_index_301[root] + quality_index_337[quality_337]
        
        return root_triad_index, bass_index[bass], seventh_index[seventh], ninth_index[ninth], eleventh_index[eleventh], thirteenth_index[thirteenth], vocab_301_index, vocab_337_index


    # This is the function we need to alter
    def transform_annotation(self, ann, duration):
        '''Apply the chord transformation.

        Parameters
        ----------
        ann : jams.Annotation
            The chord annotation

        duration : number > 0
            The target duration

        Returns
        -------
        data : dict
            data['pitch'] : np.ndarray, shape=(n, 12)
            data['root'] : np.ndarray, shape=(n, 13) or (n, 1)
            data['bass'] : np.ndarray, shape=(n, 13) or (n, 1)

            `pitch` is a binary matrix indicating pitch class
            activation at each frame.

            `root` is a one-hot matrix indicating the chord
            root's pitch class at each frame.

            `bass` is a one-hot matrix indicating the chord
            bass (lowest note) pitch class at each frame.

            If sparsely encoded, `root` and `bass` are integers
            in the range [0, 12] where 12 indicates no chord.

            If densely encoded, `root` and `bass` have an extra
            final dimension which is active when there is no chord
            sounding.
        '''
        # Construct a blank annotation with mask = 0
        intervals, chords = ann.to_interval_values()

        # Get the dtype for root/bass
        if self.sparse:
            dtype = np.int
        else:
            dtype = np.bool

        # If we don't have any labeled intervals, fill in a no-chord
        if not chords:
            intervals = np.asarray([[0, duration]])
            chords = ['N']

        # Suppress all intervals not in the encoder
        root_triads = []
        basses = []
        sevenths = []
        ninths = []
        elevenths= []
        thirteenths = []
        chords_301 = []
        chords_337 = []

        fill = 0

        for chord in chords:
            # Encode the pitches
            root_triad, bass, seventh, ninth, eleventh, thirteenth, chord_301, chord_337 = self.decompose_chord(chord)

            root_triads.append([root_triad])
            basses.append([bass])
            sevenths.append([seventh])
            ninths.append([ninth])
            elevenths.append([eleventh])
            thirteenths.append([thirteenth])
            chords_301.append([chord_301])
            chords_337.append([chord_337])

        root_triads = np.asarray(root_triads, dtype=dtype)
        basses = np.asarray(basses, dtype=dtype)
        sevenths = np.asarray(sevenths, dtype=dtype)
        ninths = np.asarray(ninths, dtype=dtype)
        elevenths = np.asarray(elevenths, dtype=dtype)
        thirteenths = np.asarray(thirteenths, dtype=dtype)
        chords_301 = np.asarray(chords_301, dtype=dtype)
        chords_337 = np.asarray(chords_337, dtype=dtype)

        target_root_triad = self.encode_intervals(duration, intervals, root_triads,
                                                  multi=False,
                                                  dtype=dtype,
                                                  fill=fill)

        target_bass = self.encode_intervals(duration, intervals, basses,
                                            multi=False,
                                            dtype=dtype,
                                            fill=fill)
        
        target_seventh = self.encode_intervals(duration, intervals, sevenths,
                                            multi=False,
                                            dtype=dtype,
                                            fill=fill)

        target_ninth = self.encode_intervals(duration, intervals, ninths,
                                            multi=False,
                                            dtype=dtype,
                                            fill=fill)

        target_eleventh = self.encode_intervals(duration, intervals, elevenths,
                                            multi=False,
                                            dtype=dtype,
                                            fill=fill)

        target_thirteenth = self.encode_intervals(duration, intervals, thirteenths,
                                            multi=False,
                                            dtype=dtype,
                                            fill=fill)
        
        target_chord_301 = self.encode_intervals(duration, intervals, chords_301,
                                            multi=False,
                                            dtype=dtype,
                                            fill=300)

        target_chord_337 = self.encode_intervals(duration, intervals, chords_337,
                                            multi=False,
                                            dtype=dtype,
                                            fill=336)

        return {'root_triad': target_root_triad,
                'bass': target_bass,
                'seventh': target_seventh,
                'ninth': target_ninth,
                'eleventh': target_eleventh,
                'thirteenth': target_thirteenth,
                'chord_301': target_chord_301,
                'chord_337': target_chord_337}

    def inverse(self, pitch, root, bass, duration=None):

        raise NotImplementedError('Chord cannot be inverted')


class SimpleChordTransformer(ChordTransformer):
    '''Simplified chord transformations.  Only pitch class activity is encoded.

    Attributes
    ----------
    name : str
        name of the transformer

    sr : number > 0
        Sampling rate of audio

    hop_length : int > 0
        Hop length for annotation frames

    See Also
    --------
    ChordTransformer
    '''
    def __init__(self, name='chord', sr=22050, hop_length=512):
        super(SimpleChordTransformer, self).__init__(name=name,
                                                     sr=sr,
                                                     hop_length=hop_length)
        # Remove the extraneous fields
        self.pop('root')
        self.pop('bass')

    def transform_annotation(self, ann, duration):
        '''Apply the chord transformation.

        Parameters
        ----------
        ann : jams.Annotation
            The chord annotation

        duration : number > 0
            The target duration

        Returns
        -------
        data : dict
            data['pitch'] : np.ndarray, shape=(n, 12)

            `pitch` is a binary matrix indicating pitch class
            activation at each frame.
        '''
        data = super(SimpleChordTransformer,
                     self).transform_annotation(ann, duration)

        data.pop('root', None)
        data.pop('bass', None)
        return data

    def inverse(self, *args, **kwargs):
        raise NotImplementedError('SimpleChord cannot be inverted')


'''A list of normalized pitch class names'''
PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


'''A mapping of chord quality encodings to their names'''
QUALITIES = {
    0b000100000000: 'min',
    0b000010000000: 'maj',
    0b000100010000: 'min',
    0b000010010000: 'maj',
    0b000100100000: 'dim',
    0b000010001000: 'aug',
    0b000100010010: 'min7',
    0b000010010001: 'maj7',
    0b000010010010: '7',
    0b000100100100: 'dim7',
    0b000100100010: 'hdim7',
    0b000100010001: 'minmaj7',
    0b000100010100: 'min6',
    0b000010010100: 'maj6',
    0b001000010000: 'sus2',
    0b000001010000: 'sus4'
}


class ChordTagTransformer(BaseTaskTransformer):
    '''Chord transformer that uses a tag-space encoding for chord labels.

    Attributes
    ----------
    name : str
        name of the transformer

    vocab : str

        A string of chord quality indicators to include:

            - '3': maj/min
            - '5': '3' + aug/dim
            - '6': '3' + '5' + maj6/min6
            - '7': '3' + '5' + '6' + 7/min7/maj7/dim7/hdim7/minmaj7
            - 's': sus2/sus4

        Note: 5 requires 3, 6 requires 5, 7 requires 6.

    sr : number > 0
        Sampling rate of audio

    hop_length : int > 0
        Hop length for annotation frames

    p_self : None, float in (0, 1), or np.ndarray [shape=(n_labels,)]
        Optional self-loop probability(ies), used for Viterbi decoding

    p_state : None or np.ndarray [shape=(n_labels,)]
        Optional marginal probability for each chord class

    p_init : None or np.ndarray [shape=(n_labels,)]
        Optional initial probability for each chord class

    Notes
    -----
    The number of chord classes (`n_labels`) depends on the vocabulary:

        - '3' => 2 + 12 * 2 = 26
        - '35' => 2 + 12 * 4 = 50
        - '356' => 2 + 12 * 6 = 74
        - '3567' => 2 + 12 * 12 = 146
        - '3567s' => 2 + 12 * 14 = 170

    See Also
    --------
    ChordTransformer
    SimpleChordTransformer
    '''
    def __init__(self, name='chord', vocab='3567s',
                 sr=22050, hop_length=512, sparse=False,
                 p_self=None, p_init=None, p_state=None):

        super(ChordTagTransformer, self).__init__(name=name,
                                                  namespace='chord',
                                                  sr=sr,
                                                  hop_length=hop_length)

        # Stringify and lowercase
        if set(vocab) - set('3567s'):
            raise ParameterError('Invalid vocabulary string: {}'.format(vocab))

        if '5' in vocab and '3' not in vocab:
            raise ParameterError('Invalid vocabulary string: {}'.format(vocab))

        if '6' in vocab and '5' not in vocab:
            raise ParameterError('Invalid vocabulary string: {}'.format(vocab))

        if '7' in vocab and '6' not in vocab:
            raise ParameterError('Invalid vocabulary string: {}'.format(vocab))

        self.vocab = vocab.lower()
        labels = self.vocabulary()
        self.sparse = sparse

        if self.sparse:
            self.encoder = LabelEncoder()
        else:
            self.encoder = LabelBinarizer()
        self.encoder.fit(labels)
        self._classes = set(self.encoder.classes_)

        self.set_transition(p_self)

        if p_init is not None:
            if len(p_init) != len(self._classes):
                raise ParameterError('Invalid p_init.shape={} for vocabulary {} size={}'.format(p_init.shape, vocab, len(self._classes)))

        self.p_init = p_init

        if p_state is not None:
            if len(p_state) != len(self._classes):
                raise ParameterError('Invalid p_state.shape={} for vocabulary {} size={}'.format(p_state.shape, vocab, len(self._classes)))

        self.p_state = p_state

        # Construct the quality mask for chord encoding
        self.mask_ = 0b000000000000
        if '3' in self.vocab:
            self.mask_ |= 0b000110000000
        if '5' in self.vocab:
            self.mask_ |= 0b000110111000
        if '6' in self.vocab:
            self.mask_ |= 0b000110010100
        if '7' in self.vocab:
            self.mask_ |= 0b000110110111
        if 's' in self.vocab:
            self.mask_ |= 0b001001010000

        if self.sparse:
            self.register('chord', [None, 1], np.int)
        else:
            self.register('chord', [None, len(self._classes)], np.bool)

    def set_transition(self, p_self):
        '''Set the transition matrix according to self-loop probabilities.

        Parameters
        ----------
        p_self : None, float in (0, 1), or np.ndarray [shape=(n_labels,)]
            Optional self-loop probability(ies), used for Viterbi decoding
        '''
        if p_self is None:
            self.transition = None
        else:
            self.transition = transition_loop(len(self._classes), p_self)

    def empty(self, duration):
        '''Empty chord annotations

        Parameters
        ----------
        duration : number
            The length (in seconds) of the empty annotation

        Returns
        -------
        ann : jams.Annotation
            A chord annotation consisting of a single `no-chord` observation.
        '''
        ann = super(ChordTagTransformer, self).empty(duration)

        ann.append(time=0,
                   duration=duration,
                   value='X', confidence=0)

        return ann

    def vocabulary(self):
        qualities = []

        if '3' in self.vocab or '5' in self.vocab:
            qualities.extend(['min', 'maj'])

        if '5' in self.vocab:
            qualities.extend(['dim', 'aug'])

        if '6' in self.vocab:
            qualities.extend(['min6', 'maj6'])

        if '7' in self.vocab:
            qualities.extend(['min7', 'maj7', '7', 'dim7', 'hdim7', 'minmaj7'])

        if 's' in self.vocab:
            qualities.extend(['sus2', 'sus4'])

        labels = ['N', 'X']

        for chord in product(PITCHES, qualities):
            labels.append('{}:{}'.format(*chord))

        return labels

    def simplify(self, chord):
        '''Simplify a chord string down to the vocabulary space'''
        # Drop inversions
        chord = re.sub(r'/.*$', r'', chord)
        # Drop any additional or suppressed tones
        chord = re.sub(r'\(.*?\)', r'', chord)
        # Drop dangling : indicators
        chord = re.sub(r':$', r'', chord)

        # Encode the chord
        root, pitches, _ = mir_eval.chord.encode(chord)

        # Build the query
        # To map the binary vector pitches down to bit masked integer,
        # we just dot against powers of 2
        P = 2**np.arange(12, dtype=int)
        query = self.mask_ & pitches[::-1].dot(P)

        if root < 0 and chord[0].upper() == 'N':
            return 'N'
        if query not in QUALITIES:
            return 'X'

        return '{}:{}'.format(PITCHES[root], QUALITIES[query])

    def transform_annotation(self, ann, duration):
        '''Transform an annotation to chord-tag encoding

        Parameters
        ----------
        ann : jams.Annotation
            The annotation to convert

        duration : number > 0
            The duration of the track

        Returns
        -------
        data : dict
            data['chord'] : np.ndarray, shape=(n, n_labels)
                A time-varying binary encoding of the chords
        '''

        intervals, values = ann.to_interval_values()

        chords = []
        for v in values:
            chords.extend(self.encoder.transform([self.simplify(v)]))

        dtype = self.fields[self.scope('chord')].dtype

        chords = np.asarray(chords)

        if self.sparse:
            chords = chords[:, np.newaxis]

        target = self.encode_intervals(duration, intervals, chords,
                                       multi=False, dtype=dtype)

        return {'chord': target}

    def inverse(self, encoded, duration=None):
        '''Inverse transformation'''

        ann = jams.Annotation(self.namespace, duration=duration)

        for start, end, value in self.decode_intervals(encoded,
                                                       duration=duration,
                                                       multi=False,
                                                       sparse=self.sparse,
                                                       transition=self.transition,
                                                       p_init=self.p_init,
                                                       p_state=self.p_state):

            # Map start:end to frames
            f_start, f_end = time_to_frames([start, end],
                                            sr=self.sr,
                                            hop_length=self.hop_length)

            # Reverse the index
            if self.sparse:
                # Compute the confidence
                if encoded.shape[1] == 1:
                    # This case is for full-confidence prediction (just the index)
                    confidence = 1.
                else:
                    confidence = np.mean(encoded[f_start:f_end+1, value])

                value_dec = self.encoder.inverse_transform(value)
            else:
                confidence = np.mean(encoded[f_start:f_end+1, np.argmax(value)])
                value_dec = self.encoder.inverse_transform(np.atleast_2d(value))

            for vd in value_dec:
                ann.append(time=start,
                           duration=end-start,
                           value=vd,
                           confidence=float(confidence))

        return ann