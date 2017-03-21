import os
import scipy.io.wavfile as wav
import numpy as np
from pipes import quote
from config import nn_config
#import matplotlib.pyplot as plt
#import wave

def convert_mp3_to_wav(filename, sample_frequency):
    
    # This statement gets the file extension for the file to make sure that it is an mp3 file before conversion begins
    ext = filename[-4:]

    if (ext != '.mp3'):
        return

    # The below statement splits the path for the file into an array of individual strings
    files = filename.split('/')

    # orig_filename now stores the name of the music file without the extension
    # orig_path stores the path of the mp3 file to be converted
    orig_filename = files[-1][0:-4]
    orig_path = filename[0:-len(files[-1])]

    # We define a variable new_path
    new_path = ''

    # The below statements define a value for new_path as the same folder in which the mp3 files lie
    if (filename[0] == '/'):
        new_path = '/'
    for i in range(len(files) - 1):
        new_path += files[i] + '/'

    # We now define two paths - one for the tmp folder for the mp3 files and one for the new_path which contains the WAV files
    # We also create directories if they don't already exist
    tmp_path = new_path + 'tmp'
    new_path += 'wave'

    if not os.path.exists(new_path):
        os.makedirs(new_path)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    # We define the file names for the newly created WAV files and the already existing(?) mp3 files
    filename_tmp = tmp_path + '/' + orig_filename + '.mp3'
    new_name = new_path + '/' + orig_filename + '.wav'

    # These lines calls LAME to resample the audio file at the standard analog frequency of 44,100 Hz and then convert it to WAV
    sample_freq_str = "{0:.1f}".format(float(sample_frequency) / 1000.0)
    cmd = 'lame -a -m m {0} {1}'.format(quote(filename), quote(filename_tmp))
    os.system(cmd)
    cmd = 'lame --decode {0} {1} --resample {2}'.format(quote(filename_tmp), quote(new_name), sample_freq_str)
    os.system(cmd)
    
    """
    '''This plots Amplitude on Y-axis and Time in seconds on X-axis'''

    # plot the converted wav file
    spf = wave.open(new_name, 'r')

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    fs = spf.getframerate()

    # To Plot x-axis in seconds you need get the frame rate and divide by size of your signal.
    # linspace function from numpy is used to create a Time Vector spaced linearly with the size of the audio file

    time = np.linspace(0, len(signal)/fs, num=len(signal))

    plt.figure(1)
    plt.title('Signal Wave...')
    plt.plot(time, signal)
    plt.show()
    """
    
    # Returns the name of the directory where all the WAV files are stored
    return new_name

# The below method converts the mp3 files in the directory to WAV files
def convert_folder_to_wav(directory, sample_rate=44100):
    # The below for loop runs through all the mp3 files and converts them to WAV
    for file in os.listdir(directory):

        # fullfilename holds the name of the directory and the file one after another
        fullfilename = directory + file

        # The below if - elif statement converts the file to WAV based on its file extension
        if file.endswith('.mp3'):
            convert_mp3_to_wav(filename=fullfilename, sample_frequency=sample_rate)


    return directory + 'wave/'


def read_wav_as_np(filename):
    # wav.read returns the sampling rate per second  (as an int) and the data (as a numpy array)
    data = wav.read(filename)
    print(data[1]) 
    np_arr = data[1].astype('float32') / 32767.0  # Normalize 16-bit input to [-1, 1] range
    # np_arr = np.array(np_arr)
    return np_arr, data[0]


def write_np_as_wav(X, sample_rate, filename):
    Xnew = X * 32767.0
    Xnew = Xnew.astype('int16')
    wav.write(filename, sample_rate, Xnew)
    return

# this returns song_np by padding it
def convert_np_audio_to_sample_blocks(song_np, block_size):  

    # Block lists initialised
    block_lists = []

    # total_samples holds the size of the numpy array
    total_samples = song_np.shape[0]
    print('total_samples = ', total_samples)

    # num_samples_so_far is used to loop through the numpy array
    num_samples_so_far = 0

    while (num_samples_so_far < total_samples):

        # Stores each block in the "block" variable
        block = song_np[num_samples_so_far:num_samples_so_far + block_size]
        '''print("block.shape[0]=", block.shape[0])'''
        if (block.shape[0] < block_size):
            # this is to add 0's in the last block if it not completely filled
            padding = np.zeros((block_size - block.shape[0],))  
            # block_size is 44100 which is fixed throughout whereas block.shape[0] for the last block is <=44100
            block = np.concatenate((block, padding))  
        block_lists.append(block)
        num_samples_so_far += block_size
    return block_lists


def convert_sample_blocks_to_np_audio(blocks):
    song_np = np.concatenate(blocks)
    '''print("songs.shape=", np.shape(song_np))'''
    return song_np


def time_blocks_to_fft_blocks(blocks_time_domain, count):
    fft_blocks = []
    #plot_block=[]   
    #amplitude = []  
    for block in blocks_time_domain:
        '''print("block=",block)'''
        # Computes the one-dimensional discrete Fourier Transform and returns the complex nD array
        # i.e The truncated or zero-padded input, transformed along the axis indicated by axis, or the last one if axis is not specified.
        fft_block = np.fft.fft(block)
        new_block = np.concatenate(
                (np.real(fft_block), np.imag(fft_block)))  # Joins a sequence of arrays along an existing axis.
        fft_blocks.append(new_block)
        #plot_block.append(fft_block)

    """
    # plots signal after fft
    timeRange = np.arange(0, 44100, 44100.0/len(plot_block))

    # Calculating the amplitude of each frequency

    for index in range(len(plot_block)):
        amplitude.append(np.sqrt(np.real(plot_block[index])**2 + np.imag(plot_block[index])**2))

    if count == 1:
        plt.title('Signal X')

    else:
        plt.title('Signal Y')

    # Plotting the DFT
    plt.plot(timeRange, amplitude)
    plt.show()
    """
    
    return fft_blocks


def fft_blocks_to_time_blocks(blocks_ft_domain):
    time_blocks = []
    for block in blocks_ft_domain:
        num_elems = block.shape[0] / 2
        real_chunk = block[0:num_elems]
        imag_chunk = block[num_elems:]
        new_block = real_chunk + 1.0j * imag_chunk
        time_block = np.fft.ifft(new_block)
        time_blocks.append(time_block)
    return time_blocks


def convert_wav_files_to_nptensor(directory, block_size, max_seq_len, out_file, max_files=20, useTimeDomain=False):
    files = []

    # If the file is already a WAV file, then the code simply stores it as it is
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            files.append(directory + file)

    # chunks_X and chunks_Y are initialized as lists
    chunks_X = []
    chunks_Y = []

    # The code takes in a maximum of 20 files and if greater, then the first twenty alone
    num_files = len(files)
    if (num_files > max_files):
        num_files = max_files

    # This loops through the indices (0 -> max_files) of the files list
    for file_idx in range(num_files):
        # Each file is stored in the variable "file"
        file = files[file_idx]

        # Prints some sort of processing message to the user, using file index and number of files
        print('Processing: ', (file_idx + 1), '/', num_files)
        print('Filename: ', file)

        X, Y = load_training_example(file, block_size, useTimeDomain=useTimeDomain)
        cur_seq = 0
        total_seq = len(X)
        print("X.shape", np.shape(X))
        print(total_seq)
        print(max_seq_len)
        while cur_seq + max_seq_len < total_seq:
            chunks_X.append(X[cur_seq:cur_seq + max_seq_len])
            chunks_Y.append(Y[cur_seq:cur_seq + max_seq_len])
            cur_seq += max_seq_len
    num_examples = len(chunks_X)  # num_examples=34 because for 1st file total_seq=133 & max_seq_len=10 so 133/10=13
    # for 2nd file total_seq=220 & max_seq_len=10 so 220/10=22. Hence num_examples=13+21=34
    num_dims_out = block_size * 2
    if (useTimeDomain):
        num_dims_out = block_size
    out_shape = (num_examples, max_seq_len, num_dims_out)
    x_data = np.zeros(out_shape)
    y_data = np.zeros(out_shape)
    for n in range(num_examples):
        for i in range(max_seq_len):
            x_data[n][i] = chunks_X[n][i]
            y_data[n][i] = chunks_Y[n][i]
        print('Saved example ', (n + 1), ' / ', num_examples)
    print('Flushing to disk...')
    # Mean across num examples and num timesteps
    mean_x = np.mean(np.mean(x_data, axis=0), axis=0)
    # STD across num examples and num timesteps
    std_x = np.sqrt(np.mean(np.mean(np.abs(x_data - mean_x) ** 2, axis=0), axis=0))
    # Clamp variance if too tiny
    std_x = np.maximum(1.0e-8, std_x)  

    '''
    x_data[:][:] -= mean_x  # Mean 0
    x_data[:][:] /= std_x  # Variance 1
    y_data[:][:] -= mean_x  # Mean 0
    y_data[:][:] /= std_x  # Variance 1
    '''
    # The above code snippet causes noise.

    np.save(out_file + '_mean', mean_x)
    np.save(out_file + '_var', std_x)
    np.save(out_file + '_x', x_data)
    np.save(out_file + '_y', y_data)
    print("x_data=", np.shape(x_data))
    print("y_data=", np.shape(y_data))
    inter_filename = out_file + '_x'
    convert_nptensor_to_wav_files_verify(x_data, num_examples, inter_filename, False)
    print('Done converting the input to the neural network to a WAV file')
    print('Done converting the WAV file to an np tensor to feed to the RNN ')


def convert_nptensor_to_wav_files_verify(tensor, indices, filename, useTimeDomain=False):
    num_seqs = tensor.shape[1]
    chunks = []
    for i in range(indices):
        for x in range(num_seqs):
            chunks.append(tensor[i][x])
    save_generated_example(filename + 'merged' + '.wav', chunks, useTimeDomain=useTimeDomain)

    """
    chunk_wav=filename+'merged.wav'

    spf = wave.open(chunk_wav, 'r')

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    fs = spf.getframerate()

    time = np.linspace(0, len(signal)/fs, num=len(signal))

    plt.figure(1)
    plt.title('Wave that goes into neural network model')
    plt.plot(time, signal)
    plt.show()
    """


def convert_nptensor_to_wav_files(tensor, indices, filename, useTimeDomain=False):
    num_seqs = tensor.shape[1]
    for i in indices:
        chunks = []
        for x in range(num_seqs):
            chunks.append(tensor[i][x])
        save_generated_example(filename + str(i) + '.wav', chunks, useTimeDomain=useTimeDomain)


def load_training_example(filename, block_size=2048, useTimeDomain=False):
    # read_wav_as_np returns data as a numpy array and the sampling rate stored in data and bitrate respectively
    data, bitrate = read_wav_as_np(filename)

    # x_t has the padded data i.e with 0's in the empty space of the last block
    x_t = convert_np_audio_to_sample_blocks(data, block_size)
    y_t = x_t[1:]
    y_t.append(np.zeros(block_size))  # Add special end block composed of all zeros
    if useTimeDomain:
        return x_t, y_t

    '''print(len(x_t))'''
    X = time_blocks_to_fft_blocks(x_t, count=1)
    Y = time_blocks_to_fft_blocks(y_t, count=2)
    print(np.shape(X))
    return X, Y


def save_generated_example(filename, generated_sequence, useTimeDomain=False, sample_frequency=44100):
    if useTimeDomain:
        time_blocks = generated_sequence
    else:
        time_blocks = fft_blocks_to_time_blocks(generated_sequence)
    song = convert_sample_blocks_to_np_audio(time_blocks)
    write_np_as_wav(song, sample_frequency, filename)
    return


def audio_unit_test(filename, filename2):
    data, bitrate = read_wav_as_np(filename)
    time_blocks = convert_np_audio_to_sample_blocks(data, 1024)
    ft_blocks = time_blocks_to_fft_blocks(time_blocks)
    time_blocks = fft_blocks_to_time_blocks(ft_blocks)
    song = convert_sample_blocks_to_np_audio(time_blocks)
    write_np_as_wav(song, bitrate, filename2)
    return
