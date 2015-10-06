"""
    SQL Queries
"""
sql_queries = {

    'Skirt': { 'query': 'SELECT clothingimagetag.ID_Image, p1_x, p1_y, p2_x, p2_y, SkirtLength, SkirtShape, SkirtPleat \
                         FROM clothingimagetag, skirtlabel \
                         WHERE clothingimagetag.ID_image=skirtlabel.ID_image \
                         AND SkirtLength!="None"',
               'points': 2,
               'groups': (0, 1, 2),         # Contains information in 'attributes_index'
               'trans' : 0                  # Refer to Group 0 for bbox transformation information
             },

    'Collar': { 'query': 'SELECT clothingimagetag.ID_Image, p1_x, p1_y, p2_x, p2_y, CollarType  \
                         FROM clothingimagetag, collarlabel \
                         WHERE clothingimagetag.ID_image=collarlabel.ID_image \
                         ',
               'points': 2,
               'groups': (4),               # Contains information in 'attributes_index'
               'trans' : 4                  # Refer to Group 0 for bbox transformation information
             },

    'Placket': { 'query': 'SELECT clothingimagetag.ID_Image, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y,  \
                         p5_x, p5_y, p6_x, p6_y,Placket1   \
                         FROM clothingimagetag, upperlabel \
                         WHERE clothingimagetag.ID_image=upperlabel.ID_image \
                         ',
               'points': 6,
               'groups': (5),               # Contains information in 'attributes_index'
               'trans' : 5                  # Refer to Group 0 for bbox transformation information
             },

    'Sleeve': { 'query': 'SELECT clothingimagetag.ID_Image, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y,  \
                         p5_x, p5_y, p6_x, p6_y, SleeveLength  \
                         FROM clothingimagetag, sleevelabel \
                         WHERE clothingimagetag.ID_image=sleevelabel.ID_image \
                         ',
               'points': 6,
               'groups': (6),               # Contains information in 'attributes_index'
               'trans' : 6                  # Refer to Group 0 for bbox transformation information
             },

    'Button': { 'query': 'SELECT clothingimagetag.ID_Image, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, \
                         p5_x, p5_y, p6_x, p6_y ButtonType \
                         FROM clothingimagetag, upperlabel \
                         WHERE clothingimagetag.ID_image=upperlabel.ID_image \
                         ',
               'points': 6,
               'groups': (7),               # Contains information in 'attributes_index'
               'trans' : 7                  # Refer to Group 0 for bbox transformation information
             },

}

"""
    Attribute Indices
"""
attributes_index = [
    # name     # (group_idx, label_idx),  # bbox transformation
                                          # (x_offset; y_offset; x_ext_factor; y_ext_factor, x_y_ratio)

    # Group 1: Skirt length
    {
    'Chang'         : [(0, 1),               (0,    0, 1.1, 1.1, 0.8)],
    'Duan'          : [(0, 2),               (0, 0.05, 1.2, 1.6, 0.8)],
    'Zhong'         : [(0, 3),               (0, 0.05, 1.2, 1.4, 0.8)],
    },

    # Group 2: Shape of Skirt
    {
    'Denglongzhuang': [(1, 1),               None],                      # 'None' means no transformation
    'Labazhuang'    : [(1, 2),               None],
    'Zhitongzhuang' : [(1, 3),               None],
    },

    # Group 3: Pleat of Skirt
    {
    'None'          : [(2, 1),               None],
    'You'           : [(2, 2),               None],
    },

    # Group 4: Collar type
    {
    'Fanling'       : [(4, 1),               (0, 0, 1.8, 1.8, 0.8)],
    'Liling'        : [(4, 2),               (0, 0, 1.8, 1.8, 0.8)],
    'None'          : [(4, 3),               (0, 0, 1.8, 1.8, 0.8)],
    },

    # Group 5: Placket type
    {
    'Duijin'        : [(5, 1),               (0, 0, 0.7, 0.5, 0.7)],
    'None'          : [(5, 2),               (0, 0, 0.7, 0.5, 0.7)],
    },

    # Group 6: Sleeve Length
    {
    'Changxiu'      : [(6, 1),           (0, -0.05, 1.0, 1.0, 0.6)],
    'Duanxiu'       : [(6, 2),            (0, 0.11, 2.0, 3.0, 0.6)],
    },

    # Group 7: Button Type
    {
    'Danpaikou'     : [(7, 1),            (0, -0.1, 0.7, 0.6, 0.8)],
    'Lalian'        : [(7, 2),            (0, -0.1, 0.7, 0.6, 0.8)],
    'None'          : [(7, 3),            (0, -0.1, 0.7, 0.6, 0.8)],
    }
]
