class Stream:
    def __init__(self, fact_kernels, channels_in, channels_out, subspace_scale, axis=1):
        #fact_kernels --> list of lists of kernels [[3,3,1],[3,1,1],...]
        self.kernels = []
        self.padding = []
        self.strides = []
        self.filters = []
        self.stream_layers = []

        n_k = len(fact_kernels)
        subspace = len(fact_kernels)-1
        assert (n_k >=2 ), 'Number of kernels must >= 2'
        assert ((fact_kernels[-1][0]==1 and fact_kernels[-1][1]==1) or (fact_kernels[-1][1]==1 and fact_kernels[-1][2]==1) or (fact_kernels[-1][0]==1 and fact_kernels[-1][2]==1)), 'Last kernel must contain at least two ones'
        for ker in fact_kernels:
            assert (len(ker)==3), "Length of each kernel must be equal to 3"
            assert ((ker[0]==ker[1]) or (ker[1]==ker[2]) or (ker[0]==ker[2])), 'Each kernel must contain two equal values'
            
            count = 0 ## count kernel values greater than 1 
            for i in ker: ## to count conv2d layers
                if i > 1:
                    count+=1
            if count > 1:
                self.stream_layers.append("conv2d")
            else:
                self.stream_layers.append("conv1d")
            if count == 3:
                assert (), 'Each kernal must contain at least 1'
            ##for padding
            if count == 0:
                self.padding.append([1,0,0])
            else:
                self.padding.append([1 if j > 1 else 0 for j in ker])
            count = 0

        if axis == 1:
            spatial_kernels = fact_kernels[:-1]
            temporal_kernel = fact_kernels[-1]
        elif axis == 2:
            spatial_kernels = fact_kernels[:-1]
            temporal_kernel = fact_kernels[-1]
        elif axis == 3:
            spatial_kernels = fact_kernels[:-1]
            temporal_kernel = fact_kernels[-1]

        # spatial convolutions (subspace)
        c = channels_in
        for s in range(subspace):
            c = Stream.get_subspace_filters( channels_in, channels_out, spatial_kernels[s], scale=subspace_scale)
            c = min(c, 512)
            #print( 's:', s, 'c:', c, 'k:', k)
            self.kernels.append( spatial_kernels[s] )
            self.filters.append( c )
            self.strides.append( [1, 1, 1] )

        # temporal convolution
        self.kernels.append( temporal_kernel )
        self.filters.append( int(channels_out*subspace_scale) )
        #print("self.filters: ", self.filters)
        self.strides.append( [1, 1, 1] )

        #print( 'axis:', axis, 'padding:', self.padding)
    
    def get_subspace_filters(features_in, features_out, kernel_size, scale=1.0):
        n0  = features_in
        ni  = features_out
        k_t = max(kernel_size) #kernel_size[0]
        k_s = max(kernel_size) #kernel_size[1]
        m = (ni * n0 * k_t * k_s * k_s)/((n0 * 1 * k_s * k_s) + (ni * k_t * 1 * 1))
        #print('m:', m, 's:', scale)
        m = max(m, k_t)
        #m = min(m, int(features_out))
        m = int(m*scale)
        #m = max(m, 3)
        m = min(m, int(features_out))
        #print ('features_in:', features_in,'features_out:', features_out,  'm:', m, 'kernel:', kernel_size)
        return m
class Factorizer:
    def __init__(self, fact_kernels, channels_in, channels_out, subspace_scale, stream_axes=[1]):
        self.streams = []

        n_streams = len(stream_axes)

        scale = 1.0/float(n_streams)
        scale = scale * subspace_scale
        c_out = channels_out #max( 3, channels_out/n_streams)

        #print 'scale:', scale, c_out
        for stream_axis in stream_axes:
            self.streams.append( Stream(fact_kernels, channels_in, c_out, subspace_scale=scale, axis=stream_axis ) )
            #print ('axes:', stream_axis, "ch_in: ",channels_in, "ch_out: ",c_out, "kernels: " ,self.streams[-1].kernels, "filters: ", self.streams[-1].filters, "padding: ", self.streams[-1].padding)

        #print (self.streams)

def test_factorizer():
    fact_ker = [[1,1,3], [3,3,1]] #[[3,1,3], [1,3,3],[1,3,1],[1,3,3],[3,3,1],[1,1,1]]
    f = Factorizer(fact_ker, 32, 32, 1.0, stream_axes=[1])
    print("fact_ker",fact_ker)
    print ("stream_layers", "kernel", "stride", "filters", "padding")
    for stream in f.streams:
        for stream_layers,kernel, stride, filters, padding in zip(stream.stream_layers, stream.kernels, stream.strides, stream.filters, stream.padding):
            print (stream_layers,kernel, stride, filters, padding)

if __name__ == '__main__':
    test_factorizer()
