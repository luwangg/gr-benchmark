#!/usr/bin/env python

import numpy
from gnuradio import gr, blocks, trellis, digital

fsm_dict = {"awgn1o2_4": (2, 4, 4,
				    (0, 2, 0, 2, 1, 3, 1, 3),
				    (0, 3, 3, 0, 1, 2, 2, 1),
				    ),
	    "rep2": (2, 1, 4, (0, 0), (0, 3)),
	    "nothing": (2, 1, 2, (0, 0), (0, 1)),
            }

class siso_decoder_priori:
    '''
    @title: Benchmark priori SISO-Decoder
    '''
    
    def __init__(self, N):
	self.N = N
    
    def run_siso_decoder(self):
	self.tb = gr.top_block()
	src = blocks.null_source(gr.sizeof_short*1)
	src_head = blocks.head(gr.sizeof_short*1, self.N)
	src_gnd = blocks.vector_source_f([0], True)  # Dummy source for a-priori input of the decoder
	
	# TX Simulation
	arg = fsm_dict['awgn1o2_4']	# Get argumnets of awgn1o2_4 fsm
	constellation = digital.constellation_qpsk() # Get constellation points
	encoder = trellis.encoder_ss(trellis.fsm(*arg), 0)
	modulator = digital.chunks_to_symbols_sc(constellation.points(), 1)
	
	# RX Simulation
	## SISO-Decoder for testing input comining (Get probabilities for encoder input words)
	siso_decoder = trellis.siso_f(trellis.fsm(*arg), self.N, 0, -1, True, False, trellis.TRELLIS_MIN_SUM)
	metrics = trellis.metrics_c(4, 1, (constellation.points()), digital.TRELLIS_EUCLIDEAN)
	
	# Sinks
	rx_sink = blocks.null_sink(gr.sizeof_float)  # Get estimated data from decoder
	
	# Connections
	self.tb.connect(src, src_head)
	self.tb.connect(src_head, encoder, modulator)
	self.tb.connect(modulator, metrics)
	self.tb.connect(src_gnd, (siso_decoder, 0))
	self.tb.connect(metrics, (siso_decoder, 1))
	self.tb.connect(siso_decoder, rx_sink)
	
	# Execute Test-Flowgraph
	self.tb.run()


class viterbi_decoder:
    '''
    @title: Benchmark Viterbi-Decoder
    '''
    
    def __init__(self, N):
	self.N = N
	
    def run_viterbi_decoder_ss(self):
	self.tb = gr.top_block()
	src = blocks.null_source(gr.sizeof_short*1)
	src_head = blocks.head(gr.sizeof_short*1, self.N)
	
	# TX Simulation
	arg = fsm_dict['awgn1o2_4']	# Get argumnets of awgn1o2_4 fsm
	constellation = digital.constellation_qpsk() # Get constellation points
	encoder = trellis.encoder_ss(trellis.fsm(*arg), 0)
	modulator = digital.chunks_to_symbols_sc(constellation.points(), 1)
	
	# RX Simulation
	## SISO-Decoder for testing input comining (Get probabilities for encoder input words)
	decoder = trellis.viterbi_s(trellis.fsm(*arg), self.N, 0, -1)
	metrics = trellis.metrics_c(4, 1, (constellation.points()), digital.TRELLIS_EUCLIDEAN)
	
	# Sinks
	rx_sink = blocks.null_sink(gr.sizeof_short)  # Get estimated data from decoder
	
	# Connections
	self.tb.connect(src, src_head)
	self.tb.connect(src_head, encoder, modulator)
	self.tb.connect(modulator, metrics)
	self.tb.connect(metrics, decoder)
	self.tb.connect(decoder, rx_sink)
	
	# Execute Test-Flowgraph
	self.tb.run()

class metrics:
    '''
    @title: Benchmark Metrics Calculation
    '''
    
    def __init__(self, N):
	self.constellation = digital.constellation_qpsk()
	self.N = N
	
    def run_metrics_qpsk(self):
	self.tb = gr.top_block()
	data = numpy.random.randint(0,9, 1024)
	constellation = digital.constellation_qpsk() # Get constellation points
	src = blocks.vector_source_c(map(int, data))
	metrics = trellis.metrics_c(4, 1, constellation.points(), digital.TRELLIS_EUCLIDEAN)
	snk = blocks.null_sink(gr.sizeof_float)
	
	self.tb.connect(src, metrics, snk)
	self.tb.run()	
	
    def run_metrics_bpsk(self):
	self.tb = gr.top_block()
	data = numpy.random.randint(0,9, 1024)
	constellation = (1, -1)
	src = blocks.vector_source_s(map(int, data))
	metrics = trellis.metrics_s(2, 1, map(int, constellation), digital.TRELLIS_EUCLIDEAN)
	snk = blocks.null_sink(gr.sizeof_float)
	
	self.tb.connect(src, metrics, snk)
	self.tb.run()	

