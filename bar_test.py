import numpy as np
import matplotlib.pyplot as plt
import config as cfg
import cv2 as cv
from matplotlib import rcParams


class BarPlotor:

    attribute_labels = None

    # Initialization, open the mysql
    def __init__(self):

        # Set font family to serif
        rcParams['font.family'] = 'serif'

        def compare(item1, item2):
            if item1[1] < item2[1]:
                return -1
            elif item1[1] > item2[1]:
                return 1
            else:
                return 0

        # Initialize the attribute_labels based on attribute_indices declared in config.py
        if self.attribute_labels is None:
            self.attribute_labels = []
            for i in range(0, len(cfg.attributes_index)):
                group = cfg.attributes_index[i]
                self.attribute_labels.append('Background')

                temp = []

                for key,values in group.iteritems():
                    # Push key and its attribute_idx
                    temp.append((key,   values[0][1]))

                temp.sort(cmp=compare)

                for item in temp:
                    self.attribute_labels.append(item[0])

    def fig2data (self,fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw ( )

        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring ( fig.canvas.tostring_rgb(), dtype=np.uint8 )
        buf.shape = ( w, h, 3 )

        return buf

    def show_label_bar(self, values, fig=None, type=int):
        n_values = np.zeros(len(self.attribute_labels), dtype=type)
        offset = 0
        for group in values:
            labels_count = len(group)

            for i in range(0, labels_count):
                n_values[offset+i] = group[i]

            offset += labels_count

        ind = np.arange(len(self.attribute_labels))
        width = 0.30

        if fig is None:
            fig = plt.figure()

        plt.clf()
        ax = fig.add_subplot(111)

        rects1 = ax.bar(ind, n_values, width,
                    color='orange',
                    error_kw = dict(elinewidth=2, ecolor='red'))

        # axes and labels
        ax.set_xlim(-width,len(ind) + width)
        ax.set_ylim(0,1)
        ax.set_ylabel('Probability')
        ax.set_title('Label Preview')
        xTickMarks = [i for i in self.attribute_labels]
        ax.set_xticks(ind)
        xtickNames = ax.set_xticklabels(xTickMarks)
        plt.setp(xtickNames, rotation=30, fontsize=10)
        plt.savefig('test.png')
        return fig

if __name__ == '__main__':

    cv.namedWindow('test')

    bplt = BarPlotor()

    values = [[0,1,0,0], [0,1,0,1],[1,0,0]]

    fig = bplt.show_label_bar(values)

    data = cv.imread('test.png',cv.IMREAD_UNCHANGED)

    fig2 = bplt.fig2data(fig)
    # plt.imshow(fig2)
    # plt.show()

    cv.imshow('test', fig2)
    cv.waitKey(0)