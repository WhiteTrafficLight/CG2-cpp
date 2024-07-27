#include "polyscope/polyscope.h"

#include "polyscope/combining_hash_functions.h"
#include "polyscope/messages.h"

#include "polyscope/file_helpers.h"
#include "polyscope/point_cloud.h"
#include "polyscope/pick.h"
#include "polyscope/curve_network.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/image_scalar_artist.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <algorithm>
#include <functional>
#include <queue>
#include "Eigen/Dense"


#include "args/args.hxx"
#include "portable-file-dialogs.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"
#include <glm/glm.hpp>

using Point = std::array<float, 3>;
using Normal = std::array<float, 3>;
using Implicit = std::array<float, 4>;

using PointList = std::vector<Point>;
using ImplicitList = std::vector<Implicit>;

int edgeTable[256] = {
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0};
int triTable[256][16] =
    {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
     {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
     {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
     {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
     {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
     {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
     {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
     {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
     {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
     {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
     {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
     {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
     {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
     {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
     {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
     {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
     {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
     {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
     {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
     {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
     {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
     {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
     {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
     {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
     {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
     {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
     {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
     {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
     {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
     {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
     {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
     {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
     {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
     {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
     {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
     {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
     {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
     {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
     {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
     {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
     {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
     {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
     {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
     {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
     {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
     {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
     {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
     {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
     {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
     {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
     {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
     {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
     {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
     {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
     {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
     {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
     {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
     {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
     {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
     {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
     {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
     {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
     {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
     {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
     {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
     {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
     {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
     {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
     {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
     {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
     {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
     {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
     {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
     {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
     {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
     {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
     {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
     {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
     {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
     {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
     {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
     {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
     {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
     {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
     {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
     {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
     {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
     {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
     {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
     {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
     {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
     {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
     {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
     {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
     {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
     {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
     {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
     {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
     {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
     {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
     {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
     {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
     {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
     {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
     {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
     {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
     {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
     {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
     {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
     {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
     {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
     {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
     {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
     {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
     {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
     {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
     {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
     {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
     {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
     {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
     {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
     {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
     {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
     {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
     {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
     {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
     {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
     {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
     {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
     {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
     {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
     {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
     {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
     {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
     {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
     {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
     {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
     {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
     {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
     {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
     {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
     {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
     {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
     {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
     {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
     {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
     {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
     {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
     {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
     {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
     {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
     {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
     {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
     {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
     {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
     {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
     {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
     {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
     {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
     {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
     {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
     {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
     {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
     {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
     {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
     {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
     {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
     {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
     {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
     {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
     {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
     {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
     {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
     {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
     {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
     {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
     {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
     {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
     {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
     {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
     {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
     {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
     {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
     {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
     {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
     {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
     {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
     {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
     {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

/**
 * Teach polyscope how to handle our datatype
 */

float adaptorF_custom_accessVector3Value(const Point &v, unsigned int ind)
{
    return v[ind];
}

void readOff(std::string const &filename, std::vector<Point> *points, std::vector<Normal> *normals = nullptr)
{
    // points->clear();
    // normals->clear();
    std::ifstream obj;
    obj.open(filename);
    if (obj.fail())
    {
        return;
    }
    std::string s;
    int a, b, c;
    obj >> s;
    obj >> a;
    obj >> b;
    obj >> c;
    if (s == "OFF")
    {
        float x, y, z;
        for (int i = 0; i < a; i++)
        {
            obj >> x;
            obj >> y;
            obj >> z;
            points->push_back(Point{
                x,
                y,
                z});
        }
    }
    else if (s == "NOFF")
    {
        float x, y, z, n1, n2, n3;
        for (int i = 0; i < a; i++)
        {
            obj >> x;
            obj >> y;
            obj >> z;
            obj >> n1;
            obj >> n2;
            obj >> n3;
            points->push_back(Point{
                x,
                y,
                z});
            normals->push_back(Normal{
                n1,
                n2,
                n3});
        }
    }

    obj.close();
}
void readOffobj(std::string const &filename, std::vector<Point> *points, std::vector<std::array<int,3>> *edges)
{
    // points->clear();
    // normals->clear();
    std::ifstream obj;
    obj.open(filename);
    if (obj.fail())
    {
        return;
    }
    std::string line;
    while(std::getline(obj,line)){
        std::istringstream iss(line);
        std::string type;
        //std::getline(obj,type);
        iss >> type;
        
        if(type == "v"){
            float vx, vy, vz;
            iss >> vx;
            iss >> vy;
            iss >> vz;
            points->push_back(Point {vx, vy, vz});
        }
        else if(type == "f"){
            int e0, e1, e2, vt0, vt1, vt2;
            char c0, c1, c2;
            iss >> e0;
            iss >> c0;
            iss >> vt0;
            iss >> e1;
            iss >> c1;
            iss >> vt1;
            iss >> e2;
            iss >> c2;
            iss >> vt2;
            edges->push_back(std::array<int,3> {e0-1,e1-1,e2-1});
        }
        
    }
    /*std::string type;
    float vx, vy, vz;
    obj >> type;
    
    obj >> vx;
        obj >> vy;
        obj >> vz;
        points->push_back(Point {vx, vy, vz});
    
    polyscope::warning(type);*/


    obj.close();
}

struct EuclideanDistance
{
    static float measure(Point const &p1, Point const &p2)
    {
        float dx = p1[0] - p2[0];
        float dy = p1[1] - p2[1];
        float dz = p1[2] - p2[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
};
struct Math
{
    static int fact(int n)
    {
        return (n == 0) || (n == 1) ? 1 : n * fact(n - 1);
    }
    static float delta(int i, int j)
    {
        return (i == j) ? 1.0 : 0.0;
    }
};
struct Bezier
{
    static float Bernstein(int m, int i, float u)
    {
        if (i < 0 || i > m)
            return 0.0;
        if (u == 0.0)
            return Math::delta(i, 0);
        if (u == 1.0)
            return Math::delta(i, m);
        float a = Math::fact(m) / Math::fact(i) / Math::fact(m - i) * std::pow(1 - u, m - i) * std::pow(u, i);
        return a;
    }
    static float Bderivative(int m, int i, float u)
    {
        // float a = Math::fact(m)/Math::fact(i)/Math::fact(m-i)*std::pow(1-u, m-i-1)*std::pow(u, i-1)*(i-m*u);
        float a = m * (Bezier::Bernstein(m - 1, i - 1, u) - Bezier::Bernstein(m - 1, i, u));
        return a;
    }
    static Point q(int m, int n, float u, float v, PointList WLS)
    {
        int t = 0;
        Point sumI{0.0, 0.0, 0.0};
        for (int i = 0; i <= m; i++)
        {
            Point sumJ{0.0, 0.0, 0.0};
            for (int j = 0; j <= n; j++)
            {
                Point sum1{Bezier::Bernstein(n, j, v) * WLS[t][0], Bezier::Bernstein(n, j, v) * WLS[t][1], Bezier::Bernstein(n, j, v) * WLS[t][2]};
                sumJ[0] += sum1[0];
                sumJ[1] += sum1[1];
                sumJ[2] += sum1[2];
                t += 1;
            }
            Point sum2{Bezier::Bernstein(m, i, u) * sumJ[0], Bezier::Bernstein(m, i, u) * sumJ[1], Bezier::Bernstein(m, i, u) * sumJ[2]};
            sumI[0] += sum2[0];
            sumI[1] += sum2[1];
            sumI[2] += sum2[2];
        }
        return sumI;
    }
    static Point q_u(int m, int n, float u, float v, PointList WLS)
    {
        int t = 0;
        Point sumI{0.0, 0.0, 0.0};
        for (int i = 0; i <= m; i++)
        {
            Point sumJ{0.0, 0.0, 0.0};
            for (int j = 0; j <= n; j++)
            {
                Point sum1{Bezier::Bernstein(n, j, v) * WLS[t][0], Bezier::Bernstein(n, j, v) * WLS[t][1], Bezier::Bernstein(n, j, v) * WLS[t][2]};
                sumJ[0] += sum1[0];
                sumJ[1] += sum1[1];
                sumJ[2] += sum1[2];
                t += 1;
            }
            Point sum2{Bezier::Bderivative(m, i, u) * sumJ[0], Bezier::Bderivative(m, i, u) * sumJ[1], Bezier::Bderivative(m, i, u) * sumJ[2]};
            sumI[0] += sum2[0];
            sumI[1] += sum2[1];
            sumI[2] += sum2[2];
        }
        return sumI;
    }
    static Point q_v(int m, int n, float u, float v, PointList WLS)
    {

        int t = 0;
        Point sumI{0.0, 0.0, 0.0};
        for (int i = 0; i <= m; i++)
        {
            Point sumJ{0.0, 0.0, 0.0};
            for (int j = 0; j <= n; j++)
            {
                Point sum1{Bezier::Bderivative(n, j, v) * WLS[t][0], Bezier::Bderivative(n, j, v) * WLS[t][1], Bezier::Bderivative(n, j, v) * WLS[t][2]};
                sumJ[0] += sum1[0];
                sumJ[1] += sum1[1];
                sumJ[2] += sum1[2];
                t += 1;
            }
            Point sum2{Bezier::Bernstein(m, i, u) * sumJ[0], Bezier::Bernstein(m, i, u) * sumJ[1], Bezier::Bernstein(m, i, u) * sumJ[2]};
            sumI[0] += sum2[0];
            sumI[1] += sum2[1];
            sumI[2] += sum2[2];
        }
        return sumI;
    }
    static std::array<float, 3> normal(int m, int n, float u, float v, PointList WLS)
    {
        Point q_u = Bezier::q_u(m, n, u, v, WLS);
        Point q_v = Bezier::q_v(m, n, u, v, WLS);
        float x = q_u[1] * q_v[2] - q_u[2] * q_v[1];
        float y = q_u[2] * q_v[0] - q_u[0] * q_v[2];
        float z = q_u[0] * q_v[1] - q_u[1] * q_v[0];
        float leng = std::sqrt(x * x + y * y + z * z);
        std::array<float, 3> normal = {x / leng * (float)0.05, y / leng * (float)0.05, z / leng * (float)0.05};
        return normal;
    }
};

/*
 * This is not yet a spatial data structure :)
 */
class SpatialDataStructure
{
public:
    SpatialDataStructure(PointList const &points)
        : m_points(points), root(nullptr)
    {
        build(points);
    }

    virtual ~SpatialDataStructure() = default;

    PointList const &getPoints() const
    {
        return m_points;
    }

    void build(PointList const &points)
    {
        std::vector<int> indices(points.size());
        std::iota(std::begin(indices), std::end(indices), 0);

        root = buildRecursive(indices.data(), (int)points.size(), 0);
    }

    virtual std::vector<std::size_t> collectInRadius(const Point &p, float radius) const
    {
        std::vector<std::size_t> result;
        collectInRadiusRecursive(p, this->root, result, radius);

        return result;
    }

    virtual std::vector<std::size_t> collectKNearest(const Point &p, unsigned int k) const
    {
        std::vector<std::size_t> result;
        std::vector<std::pair<float, int>> queue;
        collectKNearestRecursive(p, root, queue, k);
        for (unsigned int i = 0; i < k; i++)
        {
            int idx = queue[i].second;
            result.push_back(idx);
        }

        return result;
    }

private:
    PointList m_points;
    struct Node
    {
        int idx;
        Node *next[2];
        int axis;
        Node() : idx(-1), axis(-1) { next[0] = next[1] = nullptr; }
    };
    Node *root;

    Node *buildRecursive(int *indices, int npoints, int depth)
    {
        if (npoints <= 0)
            return nullptr;

        const int axis = depth % 3;
        const int mid = (npoints - 1) / 2;
        std::nth_element(indices, indices + mid, indices + npoints, [&](int lhs, int rhs)
                         { return m_points[lhs][axis] < m_points[rhs][axis]; });

        Node *node = new Node();
        node->idx = indices[mid];
        node->axis = axis;

        node->next[0] = buildRecursive(indices, mid, depth + 1);
        node->next[1] = buildRecursive(indices + mid + 1, npoints - mid - 1, depth + 1);
        return node;
    }
    void collectInRadiusRecursive(Point q, Node *node, std::vector<size_t> &result, float radius) const
    {
        if (node == nullptr)
            return;

        const Point &temp = m_points[node->idx];

        const float dist = EuclideanDistance::measure(q, temp);
        if (dist < radius)
            result.push_back(node->idx);

        const int axis = node->axis;
        const int dir = q[axis] < temp[axis] ? 0 : 1;
        collectInRadiusRecursive(q, node->next[dir], result, radius);

        const float diff = fabs(q[axis] - temp[axis]);
        if (diff < radius)
        {
            collectInRadiusRecursive(q, node->next[!dir], result, radius);
        }
    }
    void collectKNearestRecursive(Point q, Node *node, std::vector<std::pair<float, int>> &queue, std::size_t k) const
    {
        if (node == nullptr)
            return;

        const Point temp = m_points[node->idx];

        const float dist = EuclideanDistance::measure(q, temp);
        queue.push_back(std::make_pair(dist, node->idx));
        std::sort(queue.begin(), queue.end());
        if (queue.size() > k)
            queue.resize(k);
        const int axis = node->axis;
        const int dir = q[axis] < temp[axis] ? 0 : 1;
        collectKNearestRecursive(q, node->next[dir], queue, k);
        const float diff = fabs(q[axis] - temp[axis]);
        if (queue.size() < k || diff < queue.back().first)
            collectKNearestRecursive(q, node->next[!dir], queue, k);
    }
};

float weight(Point fixedPoint, Point X, float h)
{
    float d = EuclideanDistance::measure(fixedPoint, X);
    return std::pow(1 - d / h, 4) * (4 * d / h + 1);
}

// Application variables
polyscope::PointCloud *pc = nullptr;
std::unique_ptr<SpatialDataStructure> sds;
std::vector<Normal> normals;
std::unique_ptr<SpatialDataStructure> sds2;
std::unique_ptr<SpatialDataStructure> sds3;
float minX, minY, minZ, maxX, maxY, maxZ;

polyscope::PointCloud *box = nullptr;
PointList posN, negN;
polyscope::PointCloud *pN = nullptr;
polyscope::PointCloud *nN = nullptr;
ImplicitList functionVal;
ImplicitList gridVal;
PointList n_3;
polyscope::SurfaceMesh *polygon = nullptr;
polyscope::PointCloud *polygonP = nullptr;
polyscope::CurveNetwork *cub = nullptr;
polyscope::PointCloud *rayIntersect = nullptr;
std::vector<float> depth;
std::vector<float> depth_Z;
std::vector<bool> contact;
/*PointList grid;
PointList WLS;
polyscope::PointCloud *wlsp = nullptr;
polyscope::CurveNetwork *grd = nullptr;
polyscope::CurveNetwork *wls = nullptr;
polyscope::SurfaceMesh *bsf = nullptr;
polyscope::CurveNetwork *nml = nullptr;*/

float gridGernate(int Nx, int Ny, int Nz)
{   
    PointList BoundingBox;
    minX = 10000.0, minY = 10000.0, minZ = 10000.0, maxX = -10000.0, maxY = -10000.0, maxZ = -10000.0;
    for (size_t i = 0; i < sds->getPoints().size(); i++)
    {
        if (minX > sds->getPoints()[i][0])
            minX = sds->getPoints()[i][0];
        if (minY > sds->getPoints()[i][1])
            minY = sds->getPoints()[i][1];
        if (minZ > sds->getPoints()[i][2])
            minZ = sds->getPoints()[i][2];
        if (maxX < sds->getPoints()[i][0])
            maxX = sds->getPoints()[i][0];
        if (maxY < sds->getPoints()[i][1])
            maxY = sds->getPoints()[i][1];
        if (maxZ < sds->getPoints()[i][2])
            maxZ = sds->getPoints()[i][2];
    }
    minX = minX - 0.1 * (maxX - minX);
    maxX = maxX + 0.1 * (maxX - minX);
    minY = minY - 0.1 * (maxY - minY);
    maxY = maxY + 0.1 * (maxY - minY);
    minZ = minZ - 0.1 * (maxZ - minZ);
    maxZ = maxZ + 0.1 * (maxZ - minZ);
    float unitXL = (maxX - minX) / Nx, unitYL = (maxY - minY) / Ny, unitZL = (maxZ - minZ) / Nz;
    for (int i = 0; i <= Nx; i++)
    {
        for (int j = 0; j <= Ny; j++)
        {
            for (int k = 0; k <= Nz; k++)
            {
                BoundingBox.push_back({minX + unitXL * i, minY + unitYL * j, minZ + unitZL * k});
            }
        }
    }
    box = polyscope::registerPointCloud("Box", BoundingBox);
    sds3 = std::make_unique<SpatialDataStructure>(BoundingBox);
    float diagonal = EuclideanDistance::measure(Point{minX, minY, minZ}, Point{maxX, maxY, maxZ});
    polyscope::state::boundingBox = std::tuple<glm::vec3, glm::vec3>{ {minX, minY, minZ}, {maxX, maxY, maxZ} };
    return diagonal;
}

void n3(float diagonal)
{
    PointList posN, negN;
    std::vector<float> alp, alp2;
    for (size_t i = 0; i < sds->getPoints().size(); i++)
    {
        float alpha = 0.01 * diagonal;
        Point temp = sds->getPoints()[i];
        functionVal.push_back(Implicit{temp[0], temp[1], temp[2], 0.0});
        n_3.push_back(temp);
        Normal tempN = normals[i];
        Point pos = Point{temp[0] + alpha * tempN[0], temp[1] + alpha * tempN[1], temp[2] + alpha * tempN[2]};
        Point neg = Point{temp[0] - alpha * tempN[0], temp[1] - alpha * tempN[1], temp[2] - alpha * tempN[2]};
        while (!sds->collectInRadius(pos, alpha).empty())
        {
            alpha = alpha / 2.0;
        }
        pos = Point{temp[0] + alpha * tempN[0], temp[1] + alpha * tempN[1], temp[2] + alpha * tempN[2]};
        posN.push_back(pos);
        n_3.push_back(pos);
        functionVal.push_back(Implicit{pos[0], pos[1], pos[2], alpha});
        alp.push_back(alpha);
        alpha = 0.01 * diagonal;
        while (!sds->collectInRadius(neg, alpha).empty())
        {
            alpha = alpha / 2.0;
        }
        neg = Point{temp[0] - alpha * tempN[0], temp[1] - alpha * tempN[1], temp[2] - alpha * tempN[2]};
        negN.push_back(neg);
        n_3.push_back(neg);
        functionVal.push_back(Implicit{neg[0], neg[1], neg[2], -alpha});
        alp2.push_back(-alpha);
    }

    sds2 = std::make_unique<SpatialDataStructure>(n_3);
    pN = polyscope::registerPointCloud("PN", posN);
    pN->addScalarQuantity("fx", alp);
    nN = polyscope::registerPointCloud("nN", negN);
    nN->addScalarQuantity("fx", alp2);
}
float functionValue(Point fixed, float radius, float h)
{
    float ft = 0.0;
    float st = 0.0;
    //Eigen::Matrix<float, 4, 4> ft; ft.setZero();
    //Eigen::Matrix<float, 4, 1> st; st.setZero();
    float x = fixed[0], y = fixed[1], z = fixed[2];
    float w;
    std::vector<std::size_t> InRad = sds2->collectInRadius(fixed, radius);
    if (InRad.empty())
    {
        int idx = sds2->collectKNearest(fixed, 1)[0];
        if (functionVal[idx][3] < 0)
            w = -10000.0;
        else if(functionVal[idx][3] == 0)
            w = 0.0;
        else
            w = 10000.0;
    }
    else
    {
        for (size_t j = 0; j < InRad.size(); j++)
        {
            Point train = sds2->getPoints()[InRad[j]];
            float W = weight(fixed, train, h);
            //Eigen::Matrix<float, 4, 1> bx {1, train[0] , train[1], train[2]};
            ft += W, st += W * functionVal[InRad[j]][3];
            //ft += W * bx * bx.transpose(), st += W * bx * functionVal[InRad[j]][3];
        }

        float c = 1 / ft * st;
        w = c;
        //Eigen::Matrix<float,4,1> c = ft.inverse()*st;
        //Eigen::Matrix<float,1,4> Basis {1,x,y,z};
        //w = Basis * c;
    }
    return w;
}
std::array<float,3> finiteDifference(Point point, float radius, float h){
    float epsilon = (maxX-minX)/5000;
    float x = (functionValue(Point {point[0]+epsilon,point[1],point[2]},radius,h)-functionValue(point,radius,h))/epsilon;
    float y = (functionValue(Point {point[0],point[1]+epsilon,point[2]},radius,h)-functionValue(point,radius,h))/epsilon;
    float z = (functionValue(Point {point[0],point[1],point[2]+epsilon},radius,h)-functionValue(point,radius,h))/epsilon;
    float length = sqrt(x*x+y*y+z*z);
    return std::array<float,3> {x/length,y/length,z/length};
}
void ImplicitValue(float radius, float h)
{
    gridVal.clear();
    std::vector<float> fx;
    std::vector<std::array<float, 3>> Color;
    for (size_t i = 0; i < sds3->getPoints().size(); i++)
    {
        //float ft = 0.0;
        //float st = 0.0;
        // Eigen::Matrix<float, 4, 4> ft; ft.setZero();
        // Eigen::Matrix<float, 4, 1> st; st.setZero();
        Point fixed = sds3->getPoints()[i];
        float x = fixed[0], y = fixed[1], z = fixed[2];
        float w = functionValue(fixed, radius, h);
        /*std::vector<std::size_t> InRad = sds2->collectInRadius(fixed, radius);
        if (InRad.empty())
        {
            int idx = sds2->collectKNearest(fixed, 1)[0];
            if (functionVal[idx][3] <= 0)
                w = -10000.0;
            else
                w = 10000.0;
            w = 10000.0;
        }
        else
        {
            for (size_t j = 0; j < InRad.size(); j++)
            {
                Point train = sds2->getPoints()[InRad[j]];
                float W = weight(fixed, train, h);
                // Eigen::Matrix<float, 4, 1> bx {1, train[0] , train[1], train[2]};
                // Eigen::Matrix<float, 1, 1> bx{1};
                // ft += W * bx * bx.transpose(), st += W * bx * functionVal[InRad[j]][3];
                ft += W, st += W * functionVal[InRad[j]][3];
            }
            // Eigen::Matrix<float,4,1> c = ft.inverse()*st;
            // Eigen::Matrix<float,1,4> Basis {1,x,y,z};
            // w = Basis * c;
            // Eigen::Matrix<float, 1, 1> c = ft.inverse() * st;
            float c = 1 / ft * st;
            w = c;
        }*/
        // Eigen::Matrix<float, 1, 1> Basis{1};
        // Eigen::Matrix<float,1,10> Basis {1,x,y,z,x*y,x*z,y*z,x*x,y*y,z*z};
        // float w = Basis * c;

        gridVal.push_back(Implicit{x, y, z, w});
        if (w < 0.0)
        {
            Color.push_back(std::array<float, 3>{0.3, 0.8, 0.8});
        }
        else
        {
            Color.push_back(std::array<float, 3>{1.0, 1.0, 0.8});
        }
        fx.push_back(w);
    }
    box->addScalarQuantity("fx", fx);
    box->addColorQuantity("Color", Color);
}
Point VertexInterp(float isolevel, Point v1, Point v2, float w1, float w2)
{
    float mu;
    Point v;

    if (abs(isolevel - w1) < 0.00001)
        return v1;
    if (abs(isolevel - w2) < 0.00001)
        return v2;
    if (abs(w1 - w2) < 0.00001)
        return v1;

    mu = (isolevel - w1) / (w2 - w1);
    v = Point{v1[0] + mu * (v2[0] - v1[0]), v1[1] + mu * (v2[1] - v1[1]), v1[2] + mu * (v2[2] - v1[2])};
    return v;
    //v2 = Point{(v1[0] + v2[0]) / (float)2.0, (v1[1] + v2[1]) / (float)2.0, (v1[2] + v2[2]) / (float)2.0};
    //return v2;
}

void marchingCubes(int Nx, int Ny, int Nz, float radius, float h)
{
    PointList triangles;
    std::vector<std::array<float,3>> normal;
    int ntriang = 0;
    for (size_t i = 0; i < sds3->getPoints().size(); i++)
    {
        std::array<Point, 12> vertlist;
        if ((int)i % (Nz + 1) == Nz || (int)i % ((Ny + 1) * (Nz + 1)) >= (Nz + 1) * Ny || (int)i % ((Nx + 1) * (Ny + 1) * (Nz + 1)) >= (Nz + 1) * (Ny + 1) * Nx)
            continue;
        Point v0 = sds3->getPoints()[i];
        float w0 = gridVal[i][3];
        Point v1 = sds3->getPoints()[i + (Nz + 1) * (Ny + 1)];
        float w1 = gridVal[i + (Nz + 1) * (Ny + 1)][3];
        Point v2 = sds3->getPoints()[i + (Nz + 1) * (Ny + 1) + 1];
        float w2 = gridVal[i + (Nz + 1) * (Ny + 1) + 1][3];
        Point v3 = sds3->getPoints()[i + 1];
        float w3 = gridVal[i + 1][3];
        Point v4 = sds3->getPoints()[i + Nz + 1];
        float w4 = gridVal[i + Nz + 1][3];
        Point v5 = sds3->getPoints()[i + (Nz + 1) * (Ny + 1) + Nz + 1];
        float w5 = gridVal[i + (Nz + 1) * (Ny + 1) + Nz + 1][3];
        Point v6 = sds3->getPoints()[i + (Nz + 1) * (Ny + 1) + Nz + 2];
        float w6 = gridVal[i + (Nz + 1) * (Ny + 1) + Nz + 2][3];
        Point v7 = sds3->getPoints()[i + Nz + 2];
        float w7 = gridVal[i + Nz + 2][3];
        if (v0[0] == maxX || v0[1] == maxY || v0[2] == maxZ)
            continue;

        int cubeIdx = 0;
        float isolevel = 0.0;
        if (w0 < isolevel)
            cubeIdx |= 1;
        if (w1 < isolevel)
            cubeIdx |= 2;
        if (w2 < isolevel)
            cubeIdx |= 4;
        if (w3 < isolevel)
            cubeIdx |= 8;
        if (w4 < isolevel)
            cubeIdx |= 16;
        if (w5 < isolevel)
            cubeIdx |= 32;
        if (w6 < isolevel)
            cubeIdx |= 64;
        if (w7 < isolevel)
            cubeIdx |= 128;

        if (edgeTable[cubeIdx] == 0)
            continue;
        if (edgeTable[cubeIdx] & 1)
            vertlist[0] = VertexInterp(isolevel, v0, v1, w0, w1);
        if (edgeTable[cubeIdx] & 2)
            vertlist[1] = VertexInterp(isolevel, v1, v2, w1, w2);
        if (edgeTable[cubeIdx] & 4)
            vertlist[2] = VertexInterp(isolevel, v2, v3, w2, w3);
        if (edgeTable[cubeIdx] & 8)
            vertlist[3] = VertexInterp(isolevel, v3, v0, w3, w0);
        if (edgeTable[cubeIdx] & 16)
            vertlist[4] = VertexInterp(isolevel, v4, v5, w4, w5);
        if (edgeTable[cubeIdx] & 32)
            vertlist[5] = VertexInterp(isolevel, v5, v6, w5, w6);
        if (edgeTable[cubeIdx] & 64)
            vertlist[6] = VertexInterp(isolevel, v6, v7, w6, w7);
        if (edgeTable[cubeIdx] & 128)
            vertlist[7] = VertexInterp(isolevel, v7, v4, w7, w4);
        if (edgeTable[cubeIdx] & 256)
            vertlist[8] = VertexInterp(isolevel, v0, v4, w0, w4);
        if (edgeTable[cubeIdx] & 512)
            vertlist[9] = VertexInterp(isolevel, v1, v5, w1, w5);
        if (edgeTable[cubeIdx] & 1024)
            vertlist[10] = VertexInterp(isolevel, v2, v6, w2, w6);
        if (edgeTable[cubeIdx] & 2048)
            vertlist[11] = VertexInterp(isolevel, v3, v7, w3, w7);

        // polyscope::warning(std::to_string(triTable[cubeIdx][i]));
        for (int i = 0; triTable[cubeIdx][i] != -1; i += 3)
        {
            triangles.push_back(vertlist[triTable[cubeIdx][i]]);
            triangles.push_back(vertlist[triTable[cubeIdx][i + 1]]);
            triangles.push_back(vertlist[triTable[cubeIdx][i + 2]]);
            normal.push_back(finiteDifference(vertlist[triTable[cubeIdx][i]],radius,h));
            normal.push_back(finiteDifference(vertlist[triTable[cubeIdx][i+1]],radius,h));
            normal.push_back(finiteDifference(vertlist[triTable[cubeIdx][i+2]],radius,h));
            ntriang += 1;
        }
    }

    std::vector<std::array<size_t, 3>> triangleEdges;
    for (size_t i = 0; i < triangles.size() - 2; i += 3)
    {
        triangleEdges.push_back(std::array<size_t, 3>{i, i + 1, i + 2});
    }
    polygon = polyscope::registerSurfaceMesh("polygon", triangles, triangleEdges);
    polygon->addVertexVectorQuantity("normals",normal);
    // polygonP = polyscope::registerPointCloud("test2",triangles);
}
Eigen::MatrixXf PseudoInverse(Eigen::MatrixXf matrix) {
   Eigen::JacobiSVD< Eigen::MatrixXf > svd( matrix, Eigen::ComputeThinU | Eigen::ComputeThinV );
   float tolerance = 1.0e-6f * float(std::max(matrix.rows(), matrix.cols())) * svd.singularValues().array().abs()(0);
   return svd.matrixV()
         * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal()
         * svd.matrixU().adjoint();
}
void EmarchingCubes(int Nx, int Ny, int Nz, float radius, float h)
{
    PointList triangles;
    std::vector<std::array<float,3>> normal;
    
    for (size_t i = 0; i < sds3->getPoints().size(); i++)
    {
        int ntriang = 0;
        std::array<Point, 12> vertlist;
        if ((int)i % (Nz + 1) == Nz || (int)i % ((Ny + 1) * (Nz + 1)) >= (Nz + 1) * Ny || (int)i % ((Nx + 1) * (Ny + 1) * (Nz + 1)) >= (Nz + 1) * (Ny + 1) * Nx)
            continue;
        Point v0 = sds3->getPoints()[i];
        float w0 = gridVal[i][3];
        Point v1 = sds3->getPoints()[i + (Nz + 1) * (Ny + 1)];
        float w1 = gridVal[i + (Nz + 1) * (Ny + 1)][3];
        Point v2 = sds3->getPoints()[i + (Nz + 1) * (Ny + 1) + 1];
        float w2 = gridVal[i + (Nz + 1) * (Ny + 1) + 1][3];
        Point v3 = sds3->getPoints()[i + 1];
        float w3 = gridVal[i + 1][3];
        Point v4 = sds3->getPoints()[i + Nz + 1];
        float w4 = gridVal[i + Nz + 1][3];
        Point v5 = sds3->getPoints()[i + (Nz + 1) * (Ny + 1) + Nz + 1];
        float w5 = gridVal[i + (Nz + 1) * (Ny + 1) + Nz + 1][3];
        Point v6 = sds3->getPoints()[i + (Nz + 1) * (Ny + 1) + Nz + 2];
        float w6 = gridVal[i + (Nz + 1) * (Ny + 1) + Nz + 2][3];
        Point v7 = sds3->getPoints()[i + Nz + 2];
        float w7 = gridVal[i + Nz + 2][3];
        if (v0[0] == maxX || v0[1] == maxY || v0[2] == maxZ)
            continue;
        // if(isnan(w0)||isnan(w1)||isnan(w2)||isnan(w3)||isnan(w4)||isnan(w5)||isnan(w6)||isnan(w7)) continue;
        int cubeIdx = 0;
        float isolevel = 0.0;
        if (w0 < isolevel)
            cubeIdx |= 1;
        if (w1 < isolevel)
            cubeIdx |= 2;
        if (w2 < isolevel)
            cubeIdx |= 4;
        if (w3 < isolevel)
            cubeIdx |= 8;
        if (w4 < isolevel)
            cubeIdx |= 16;
        if (w5 < isolevel)
            cubeIdx |= 32;
        if (w6 < isolevel)
            cubeIdx |= 64;
        if (w7 < isolevel)
            cubeIdx |= 128;

        if (edgeTable[cubeIdx] == 0)
            continue;
        if (edgeTable[cubeIdx] & 1)
            vertlist[0] = VertexInterp(isolevel, v0, v1, w0, w1);
        if (edgeTable[cubeIdx] & 2)
            vertlist[1] = VertexInterp(isolevel, v1, v2, w1, w2);
        if (edgeTable[cubeIdx] & 4)
            vertlist[2] = VertexInterp(isolevel, v2, v3, w2, w3);
        if (edgeTable[cubeIdx] & 8)
            vertlist[3] = VertexInterp(isolevel, v3, v0, w3, w0);
        if (edgeTable[cubeIdx] & 16)
            vertlist[4] = VertexInterp(isolevel, v4, v5, w4, w5);
        if (edgeTable[cubeIdx] & 32)
            vertlist[5] = VertexInterp(isolevel, v5, v6, w5, w6);
        if (edgeTable[cubeIdx] & 64)
            vertlist[6] = VertexInterp(isolevel, v6, v7, w6, w7);
        if (edgeTable[cubeIdx] & 128)
            vertlist[7] = VertexInterp(isolevel, v7, v4, w7, w4);
        if (edgeTable[cubeIdx] & 256)
            vertlist[8] = VertexInterp(isolevel, v0, v4, w0, w4);
        if (edgeTable[cubeIdx] & 512)
            vertlist[9] = VertexInterp(isolevel, v1, v5, w1, w5);
        if (edgeTable[cubeIdx] & 1024)
            vertlist[10] = VertexInterp(isolevel, v2, v6, w2, w6);
        if (edgeTable[cubeIdx] & 2048)
            vertlist[11] = VertexInterp(isolevel, v3, v7, w3, w7);

        PointList SurfacePoints;
        std::vector<std::array<float,3>> SurfaceNormals;
        for (int i = 0; triTable[cubeIdx][i] != -1; i += 3)
        {
            //triangles.push_back(vertlist[triTable[cubeIdx][i]]);
            //triangles.push_back(vertlist[triTable[cubeIdx][i + 1]]);
            //triangles.push_back(vertlist[triTable[cubeIdx][i + 2]]);
            SurfacePoints.push_back(vertlist[triTable[cubeIdx][i]]);
            SurfacePoints.push_back(vertlist[triTable[cubeIdx][i + 1]]);
            SurfacePoints.push_back(vertlist[triTable[cubeIdx][i + 2]]);
            //normal.push_back(finiteDifference(vertlist[triTable[cubeIdx][i]],radius,h));
            //normal.push_back(finiteDifference(vertlist[triTable[cubeIdx][i+1]],radius,h));
            //normal.push_back(finiteDifference(vertlist[triTable[cubeIdx][i+2]],radius,h));
            SurfaceNormals.push_back(finiteDifference(vertlist[triTable[cubeIdx][i]],radius,h));
            SurfaceNormals.push_back(finiteDifference(vertlist[triTable[cubeIdx][i+1]],radius,h));
            SurfaceNormals.push_back(finiteDifference(vertlist[triTable[cubeIdx][i+2]],radius,h));
            

            ntriang += 1;
        }
        int feature = 0;  // 0 none 1 edge 2 corner
        float theta = 1.0;
        float phi = -1.0;
        
        const int nSamples = SurfacePoints.size();
        std::array<int,2> angleLarge;
        if(ntriang > 0){
            for(int i = 0; i<nSamples-1; i++){
                for(int j = i+1; j<nSamples; j++){
                    float newTheta = SurfaceNormals[i][0]*SurfaceNormals[j][0]+SurfaceNormals[i][1]*SurfaceNormals[j][1]+SurfaceNormals[i][2]*SurfaceNormals[j][2];
                    if(newTheta<theta){
                        theta = newTheta;
                        angleLarge[0] = i; angleLarge[1] = j; 
                    }
                    
                }
            }
            if(theta<0.0){
                feature = 1;
                float x = SurfaceNormals[angleLarge[0]][1]*SurfaceNormals[angleLarge[1]][2] - SurfaceNormals[angleLarge[0]][2]*SurfaceNormals[angleLarge[1]][1];
                float y = SurfaceNormals[angleLarge[0]][2]*SurfaceNormals[angleLarge[1]][0] - SurfaceNormals[angleLarge[0]][0]*SurfaceNormals[angleLarge[1]][2];
                float z = SurfaceNormals[angleLarge[0]][0]*SurfaceNormals[angleLarge[1]][1] - SurfaceNormals[angleLarge[0]][1]*SurfaceNormals[angleLarge[1]][0];
                for(int i = 0; i<nSamples; i++){
                    float newPhi = x*SurfaceNormals[i][0]+y*SurfaceNormals[i][1]+z*SurfaceNormals[i][2];
                    if(newPhi>phi) phi = newPhi;
                }
                if(phi>0.7) feature = 2;
            }
            if(feature == 1){
                Eigen::MatrixXf N(nSamples,3);
                Eigen::MatrixXf S(nSamples,3);
                Eigen::MatrixXf NS(3,3);
                for(int i = 0; i<nSamples ; i++){
                    N(i,0) = SurfaceNormals[i][0];
                    N(i,1) = SurfaceNormals[i][1];
                    N(i,2) = SurfaceNormals[i][2];                    
                }
                for(int i = 0; i<nSamples ; i++){
                    S(i,0) = SurfacePoints[i][0];
                    S(i,1) = SurfacePoints[i][1];
                    S(i,2) = SurfacePoints[i][2];                    
                }
                NS = N.transpose()*S;
                Eigen::MatrixXf P(nSamples,3);
                //P = Nt.transpose().inverse();
                //P = N.transpose().completeOrthogonalDecomposition().pseudoInverse()*NS;
                //P = ((Nt.transpose()*Nt).inverse()*Nt.transpose()).transpose();
                P = PseudoInverse(N.transpose())*NS; 
                
                for(int i = 0; i<nSamples; i++){
                    triangles.push_back(Point {P(i,0),P(i,1),P(i,2)});
                    normal.push_back(finiteDifference(Point {P(i,0),P(i,1),P(i,2)}, radius, h));
                }
            }
            else{
                for (int i = 0; triTable[cubeIdx][i] != -1; i += 3){
                    triangles.push_back(vertlist[triTable[cubeIdx][i]]);
                    triangles.push_back(vertlist[triTable[cubeIdx][i + 1]]);
                    triangles.push_back(vertlist[triTable[cubeIdx][i + 2]]);
            
                    normal.push_back(finiteDifference(vertlist[triTable[cubeIdx][i]],radius,h));
                    normal.push_back(finiteDifference(vertlist[triTable[cubeIdx][i+1]],radius,h));
                    normal.push_back(finiteDifference(vertlist[triTable[cubeIdx][i+2]],radius,h));
                }
            }    

        }
        
    }

    std::vector<std::array<size_t, 3>> triangleEdges;
    for (size_t i = 0; i < triangles.size() - 2; i += 3)
    {
        triangleEdges.push_back(std::array<size_t, 3>{i, i + 1, i + 2});
    }
    polygon = polyscope::registerSurfaceMesh("polygon", triangles, triangleEdges);
    polygon->addVertexVectorQuantity("normals",normal);
    //polygonP = polyscope::registerPointCloud("test2",triangles);
}

void rayTracing(int px, int py, float radius, float h){
    PointList intersect;
    depth.clear();
    depth_Z.clear();
    contact.clear();
    int windowWidth = polyscope::view::windowWidth;
    int windowHeight = polyscope::view::windowHeight;
    float maxL=3000.0;
    
    for(int i=0; i<px ; i++){
        for(int j=0; j<py ; j++){
            
            glm::vec2 temp = glm::vec2((float)windowWidth/(float)px/2.0f + (float)windowWidth/(float)px*(float)i,(float)windowHeight/(float)py/2.0f + (float)windowHeight/(float)py*(float)j);
            glm::vec3 tempRay = polyscope::view::screenCoordsToWorldRay(temp);
            glm::vec3 CameraPos = polyscope::view::getCameraWorldPosition();
            int sign = 1;
            glm::vec3 rayM;
            float s = 10.0;
            glm::vec3 start;
            int count = 50;
            for(int t = 50;sign == 1 && t<300 ;t++){
                rayM = CameraPos+tempRay*s*(float)t;
                sign *= (functionValue(Point {rayM.x,rayM.y,rayM.z},radius,h)>0) ? 1 : -1 ;
                if(sign == -1) start = CameraPos+tempRay*s*(float)(t-1); 
                count++;
            }
            while(functionValue(Point {rayM.x,rayM.y,rayM.z},radius,h) < -5.0){
                s *= 0.5;
                rayM = start+tempRay*s;
            }
            if(count != 300){
                intersect.push_back(Point {rayM.x, rayM.y, rayM.z});
                depth.push_back(functionValue(Point {rayM.x,rayM.y,rayM.z},radius,h));
                contact.push_back(true);
            }
            else contact.push_back(false);

            float zdistance= glm::dot(rayM-CameraPos,polyscope::view::screenCoordsToWorldRay(glm::vec2((float)windowWidth/2.0f,(float)windowHeight/2.0f)));
            depth_Z.push_back(zdistance/3000.0f);
                                

        }
    }
    rayIntersect = polyscope::registerPointCloud("rayIntersect",intersect);
    rayIntersect->addScalarQuantity("fx",depth);
 
    
    return;
}

void showCube(int i, int Nx, int Ny, int Nz)
{
    Point v0 = sds3->getPoints()[i];
    Point v1 = sds3->getPoints()[i + (Nz + 1) * (Ny + 1)];
    Point v2 = sds3->getPoints()[i + (Nz + 1) * (Ny + 1) + 1];
    Point v3 = sds3->getPoints()[i + 1];
    Point v4 = sds3->getPoints()[i + Nz + 1];
    Point v5 = sds3->getPoints()[i + (Nz + 1) * (Ny + 1) + Nz + 1];
    Point v6 = sds3->getPoints()[i + (Nz + 1) * (Ny + 1) + Nz + 2];
    Point v7 = sds3->getPoints()[i + Nz + 2];
    if (i % (Nz + 1) == Nz || i % ((Ny + 1) * (Nz + 1)) >= (Nz + 1) * Ny || i % ((Nx + 1) * (Ny + 1) * (Nz + 1)) >= (Nz + 1) * (Ny + 1) * Nx)
        return;
    PointList cubePts;
    cubePts.push_back(v0);
    cubePts.push_back(v1);
    cubePts.push_back(v2);
    cubePts.push_back(v3);
    cubePts.push_back(v4);
    cubePts.push_back(v5);
    cubePts.push_back(v6);
    cubePts.push_back(v7);
    std::vector<std::array<int, 2>> edge;
    edge.push_back({0, 1});
    edge.push_back({1, 2});
    edge.push_back({2, 3});
    edge.push_back({3, 0});
    edge.push_back({4, 5});
    edge.push_back({5, 6});
    edge.push_back({6, 7});
    edge.push_back({7, 4});
    edge.push_back({0, 4});
    edge.push_back({1, 5});
    edge.push_back({2, 6});
    edge.push_back({3, 7});

    cub = polyscope::registerCurveNetwork("Cube", cubePts, edge);
}
float areaTri(Point p0, Point p1, Point p2){
    glm::vec3 l0 = glm::vec3(p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]);
    glm::vec3 l1 = glm::vec3(p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]);
    glm::vec3 a = glm::cross(l0,l1);
    float area = 0.5*sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
    return area;
}
float cotAlpha(Point p0, Point p1, Point alpha){
    glm::vec3 l0 = glm::vec3(p0[0]-alpha[0], p0[1]-alpha[1], p0[2]-alpha[2]);
    glm::vec3 l1 = glm::vec3(p1[0]-alpha[0], p1[1]-alpha[1], p1[2]-alpha[2]);
    //float angle = glm::acos(glm::dot(glm::normalize(l0),glm::normalize(l1)));
    //return cos(angle)/sin(angle);
    glm::vec3 cross = glm::cross(l0,l1);
    float cot = glm::dot(l0,l1)/sqrt(cross.x*cross.x+cross.y*cross.y+cross.z*cross.z);
    return cot;
}
/*float circumcentric(Point c, Point p0, Point p1){
    glm::vec3 v0 = glm::vec3(p0[0]-c[0], p0[1]-c[1], p0[2]-c[2]);
    glm::vec3 v1 = glm::vec3(p1[0]-c[0], p1[1]-c[1], p1[2]-c[2]);
    float v00 = glm.dot(v0,v0);
    float v11 = glm.dot(v1,v1);
    float v01 = glm.dot(v0,v1);
    float b = 0.5f/(v00*v11-v01*v01);
    float k0 = b*v11*(v00-v01);
    float k1 = b*v00*(v11-v01);
    float x = c.x+k0*v0.x+k1*v1.x;
    float y = c.y+k0*v0.y+k1*v1.y;
    float z = c.z+k0*v0.z+k1*v1.z;
    Point centre = Point {x,y,z};
    glm::vec3 vc = glm::vec3(x-c[0], y-c[1], z-c[2]);
    return 0.0f;


}*/
/*PointList LaplacianSmoothing(std::vector<Point> points, std::vector<std::array<int,3>> edges, int iteration, float learningRate){
    std::vector<Point> S = points;
    Eigen::MatrixXf N(points.size(), points.size());
    N.setZero();  
    for(int i = 0 ; i < (int) edges.size(); i++){
        N(edges[i][0], edges[i][1]) = 1;
        N(edges[i][0], edges[i][2]) = 1;
        N(edges[i][1], edges[i][0]) = 1;
        N(edges[i][1], edges[i][2]) = 1;
        N(edges[i][2], edges[i][0]) = 1;
        N(edges[i][2], edges[i][1]) = 1;
    }
    for(int iter = 0 ; iter<iteration ; iter++){
    for(int i = 0; i < (int)points.size(); i++){
        int cnt = 0;
        float s0 = 0.0;
        float s1 = 0.0;
        float s2 = 0.0;
        for(int j = 0; j < (int)points.size(); j++){
            if(N(i,j) == 0) continue;
            cnt++;
            s0 += S[j][0];
            s1 += S[j][1];
            s2 += S[j][2];
        }
        s0 = s0/(float)cnt; s1 = s1/(float)cnt; s2 = s2/(float)cnt;
        S.at(i)[0] = learningRate*(s0-S[i][0])+S[i][0];
        S.at(i)[1] = learningRate*(s1-S[i][1])+S[i][1];
        S.at(i)[2] = learningRate*(s2-S[i][2])+S[i][2];
    }
    }

    return S;
}*/
PointList LaplacianSmoothing(std::vector<Point> points, std::vector<std::array<int,3>> edges, int iteration, float h){
    Eigen::MatrixXf P(points.size(),3);
    Eigen::MatrixXf M(points.size(), points.size());
    Eigen::MatrixXf L(points.size(), points.size());
    M.setZero();
    L.setZero();
    for(int i = 0 ; i < (int) points.size(); i++){
        P(i,0) = points[i][0];
        P(i,1) = points[i][1];
        P(i,2) = points[i][2];
    }  
    for(int i = 0 ; i < (int) edges.size(); i++){
        L(edges[i][0], edges[i][1]) = 1;
        L(edges[i][0], edges[i][2]) = 1;
        L(edges[i][1], edges[i][0]) = 1;
        L(edges[i][1], edges[i][2]) = 1;
        L(edges[i][2], edges[i][0]) = 1;
        L(edges[i][2], edges[i][1]) = 1;
    }
    for(int i = 0 ; i < (int) points.size() ; i++){
        for(int j = 0; j < (int) points.size(); j++){
            if(i == j) continue;
            L(i,i) += -L(i,j);
        }
    }
    for(int i = 0 ;  i < (int)points.size() ; i++){
        M(i,i) = -1.0f/(float) L(i,i);
    }
    for(int iter = 0 ; iter<iteration ; iter++){
        P = P+M*L*P*h;
    }
    PointList S;
    for(int i = 0 ; i < (int)points.size() ; i++){
        S.push_back(Point {P(i,0), P(i,1), P(i,2)});
    }

    return S;


}
/*PointList cotLaplacianSmoothing(std::vector<Point> points, std::vector<std::array<int,3>> edges, int iteration, float h, bool EorI){
    Eigen::MatrixXf P(points.size(), 3);
    for(int i = 0 ; i < (int) points.size(); i++){
        P(i,0) = points[i][0];
        P(i,1) = points[i][1];
        P(i,2) = points[i][2];
    }
    Eigen::MatrixXf M(points.size(), points.size());
    Eigen::MatrixXf L(points.size(), points.size());
    M.setZero();
    L.setZero();      
    for(int i = 0 ; i < (int) edges.size(); i++){
        float area = areaTri(points[edges[i][0]],points[edges[i][1]],points[edges[i][2]])/3.0f;
        float cot0 = 0.5*cotAlpha(points[edges[i][1]],points[edges[i][2]],points[edges[i][0]]);
        float cot1 = 0.5*cotAlpha(points[edges[i][0]],points[edges[i][2]],points[edges[i][1]]);
        float cot2 = 0.5*cotAlpha(points[edges[i][0]],points[edges[i][1]],points[edges[i][2]]);
        M(edges[i][0], edges[i][0]) += area;
        M(edges[i][1], edges[i][1]) += area;
        M(edges[i][2], edges[i][2]) += area;
        L(edges[i][0], edges[i][1]) += cot2;
        L(edges[i][1], edges[i][0]) += cot2;
        L(edges[i][0], edges[i][2]) += cot1;
        L(edges[i][2], edges[i][0]) += cot1;
        L(edges[i][1], edges[i][2]) += cot0;
        L(edges[i][2], edges[i][1]) += cot0;
    }
    for(int i = 0 ; i < (int)points.size(); i++){
        M(i,i) = 1.0f/M(i,i);
    }
    for(int i = 0; i < (int)points.size(); i++){
        for(int j = 0; j < (int)points.size(); j++){
            if(i == j) continue;
            L(i,i) += -L(i,j);
        }
    }
    //polyscope::warning(std::to_string(M(0,0)));
    //polyscope::warning(std::to_string(3.0f/(areaTri(points[0],points[1],points[2])+areaTri(points[0],points[1],points[3])+areaTri(points[0],points[2],points[6])+areaTri(points[0],points[3],points[4])+areaTri(points[0],points[4],points[5])+areaTri(points[0],points[5],points[6]))));
    if(EorI){
        for(int iter = 0 ; iter<iteration ; iter++){
           P = P+M*L*P*h;
        }
    }
    else{
        for(int iter = 0 ; iter<iteration ; iter++){
           P = P+M*L*(P+M*L*P*h);
        }
    }
    PointList S;
    for(int i = 0 ; i < (int)points.size() ; i++){
        S.push_back(Point {P(i,0), P(i,1), P(i,2)});
    }

    return S;


}*/
PointList cotLaplacianSmoothing(std::vector<Point> points, std::vector<std::array<int,3>> edges, int iteration, float h, bool EorI){
    Eigen::MatrixXf P(points.size(), 3);
    for(int i = 0 ; i < (int) points.size(); i++){
        P(i,0) = points[i][0];
        P(i,1) = points[i][1];
        P(i,2) = points[i][2];
    }
    
    if(!EorI){
        for(int iter = 0 ; iter<iteration ; iter++){
           Eigen::MatrixXf M(points.size(), points.size());
           Eigen::MatrixXf L(points.size(), points.size());
           M.setZero();
           L.setZero();      
           for(int i = 0 ; i < (int) edges.size(); i++){
                Point p0 = Point {P(edges[i][0],0),P(edges[i][0],1),P(edges[i][0],2)};
                Point p1 = Point {P(edges[i][1],0),P(edges[i][1],1),P(edges[i][1],2)};
                Point p2 = Point {P(edges[i][2],0),P(edges[i][2],1),P(edges[i][2],2)};
                float area = areaTri(p0,p1,p2)/3.0f;
                float cot0 = 0.5*cotAlpha(p1,p2,p0);
                float cot1 = 0.5*cotAlpha(p0,p2,p1);
                float cot2 = 0.5*cotAlpha(p0,p1,p2);
                M(edges[i][0], edges[i][0]) += area;
                M(edges[i][1], edges[i][1]) += area;
                M(edges[i][2], edges[i][2]) += area;
                L(edges[i][0], edges[i][1]) += cot2;
                L(edges[i][1], edges[i][0]) += cot2;
                L(edges[i][0], edges[i][2]) += cot1;
                L(edges[i][2], edges[i][0]) += cot1;
                L(edges[i][1], edges[i][2]) += cot0;
                L(edges[i][2], edges[i][1]) += cot0;
            }
            for(int i = 0 ; i < (int)points.size(); i++){
                M(i,i) = 1.0f/M(i,i);
            }
            for(int i = 0; i < (int)points.size(); i++){
                for(int j = 0; j < (int)points.size(); j++){
                    if(i == j) continue;
                    L(i,i) += -L(i,j);
                }
            } 
           P = P+M*L*P*h;
        }
    }
    else{
        for(int iter = 0 ; iter<iteration ; iter++){
           Eigen::MatrixXf M(points.size(), points.size());
           Eigen::MatrixXf L(points.size(), points.size());
           M.setZero();
           L.setZero();      
           for(int i = 0 ; i < (int) edges.size(); i++){
                Point p0 = Point {P(edges[i][0],0),P(edges[i][0],1),P(edges[i][0],2)};
                Point p1 = Point {P(edges[i][1],0),P(edges[i][1],1),P(edges[i][1],2)};
                Point p2 = Point {P(edges[i][2],0),P(edges[i][2],1),P(edges[i][2],2)};
                float area = areaTri(p0,p1,p2)/3.0f;
                float cot0 = 0.5*cotAlpha(p1,p2,p0);
                float cot1 = 0.5*cotAlpha(p0,p2,p1);
                float cot2 = 0.5*cotAlpha(p0,p1,p2);
                M(edges[i][0], edges[i][0]) += area;
                M(edges[i][1], edges[i][1]) += area;
                M(edges[i][2], edges[i][2]) += area;
                L(edges[i][0], edges[i][1]) += cot2;
                L(edges[i][1], edges[i][0]) += cot2;
                L(edges[i][0], edges[i][2]) += cot1;
                L(edges[i][2], edges[i][0]) += cot1;
                L(edges[i][1], edges[i][2]) += cot0;
                L(edges[i][2], edges[i][1]) += cot0;
            }
            for(int i = 0 ; i < (int)points.size(); i++){
                M(i,i) = 1.0f/M(i,i);
            }
            for(int i = 0; i < (int)points.size(); i++){
                for(int j = 0; j < (int)points.size(); j++){
                    if(i == j) continue;
                    L(i,i) += -L(i,j);
                }
            }
            Eigen::MatrixXf P2(points.size(),3);
            P2 = P+M*L*P*h;
            Eigen::MatrixXf M2(points.size(), points.size());
            Eigen::MatrixXf L2(points.size(), points.size());
            M2.setZero();
            L2.setZero();      
            for(int i = 0 ; i < (int) edges.size(); i++){
                Point p0 = Point {P2(edges[i][0],0),P2(edges[i][0],1),P2(edges[i][0],2)};
                Point p1 = Point {P2(edges[i][1],0),P2(edges[i][1],1),P2(edges[i][1],2)};
                Point p2 = Point {P2(edges[i][2],0),P2(edges[i][2],1),P2(edges[i][2],2)};
                float area = areaTri(p0,p1,p2)/3.0f;
                float cot0 = 0.5*cotAlpha(p1,p2,p0);
                float cot1 = 0.5*cotAlpha(p0,p2,p1);
                float cot2 = 0.5*cotAlpha(p0,p1,p2);
                M2(edges[i][0], edges[i][0]) += area;
                M2(edges[i][1], edges[i][1]) += area;
                M2(edges[i][2], edges[i][2]) += area;
                L2(edges[i][0], edges[i][1]) += cot2;
                L2(edges[i][1], edges[i][0]) += cot2;
                L2(edges[i][0], edges[i][2]) += cot1;
                L2(edges[i][2], edges[i][0]) += cot1;
                L2(edges[i][1], edges[i][2]) += cot0;
                L2(edges[i][2], edges[i][1]) += cot0;
            }
            for(int i = 0 ; i < (int)points.size(); i++){
                M2(i,i) = 1.0f/M2(i,i);
            }
            for(int i = 0; i < (int)points.size(); i++){
                for(int j = 0; j < (int)points.size(); j++){
                    if(i == j) continue;
                    L2(i,i) += -L(i,j);
                }
            }
            P = P+M2*L2*P2*h;
        }
    }
    
    PointList S;
    for(int i = 0 ; i < (int)points.size() ; i++){
        S.push_back(Point {P(i,0), P(i,1), P(i,2)});
    }

    return S;


}
std::vector<Point> points;
std::vector<std::array<int,3>> edges;

void callback()
{
    static bool BoxVis = false;
    static int Nx = 20;
    static int Ny = 10;
    static int Nz = 20;
    static float radius = 0.05;
    static bool nVis = false;
    static bool pVis = false;
    static float diagonal;
    //static float h;
    static int cube = 0;
    if (ImGui::Button("Load Off"))
    {
        //auto paths = pfd::open_file("Load Off", "", std::vector<std::string>{"point data (*.off)", "*.off"}, pfd::opt::none).result();
        auto paths2 = pfd::open_file("Load Off", "", std::vector<std::string>{"mesh data (*.obj)", "*.obj"}, pfd::opt::none).result();
        if (!paths2.empty())
        {
            //std::filesystem::path path(paths[0]);
            std::filesystem::path path2(paths2[0]);
            /*if (path.extension() == ".off")
            {
                // Read the point cloud
                std::vector<Point> points;
                normals.clear();
                readOff(path.string(), &points, &normals);

                // Create the polyscope geometry
                pc = polyscope::registerPointCloud("Points", points);
                if (!normals.empty())
                    pc->addVectorQuantity("normals", normals);

                // Build spatial data structure

                sds = std::make_unique<SpatialDataStructure>(points);
                diagonal = gridGernate(Nx, Ny, Nz);
                
                h = diagonal/10.0;
                box->setEnabled(BoxVis);
                n3(diagonal);
                pN->setEnabled(nVis);
                nN->setEnabled(nVis);
                ImplicitValue(radius, h);
                marchingCubes(Nx, Ny, Nz, radius, h);
                polygon->setEnabled(pVis);
            }*/
            if (path2.extension() == ".obj"){
                points.clear();
                edges.clear();
                readOffobj(path2.string(), &points, &edges);
                
                
                pc = polyscope::registerPointCloud("Points",points);
                polygon = polyscope::registerSurfaceMesh("Mesh",points,edges);
                sds = std::make_unique<SpatialDataStructure>(points);
            }
        }
    }
    static int iteration = 0;
    static float h = 0.005;
    static bool EorI = false;
    ImGui::SliderInt("iteration", &iteration, 0 , 100);
    ImGui::SliderFloat("step size", &h, 0.0, 0.01);
    ImGui::SliderFloat("step size L", &h, 0.0, 1.0);
    ImGui::Checkbox("Explicit or Implicit", &EorI);
    if (ImGui::Button("Uniform Laplacian")){
        polygon = polyscope::registerSurfaceMesh("Mesh",LaplacianSmoothing(points,edges,iteration,h),edges);
    }
    if (ImGui::Button("cotangent Laplacian")){
        polygon = polyscope::registerSurfaceMesh("Mesh",cotLaplacianSmoothing(points,edges,iteration,h,EorI),edges);
    }


    /*if (ImGui::Checkbox("BoundingBox", &BoxVis))
    {
        box->setEnabled(BoxVis);
    }
    if (ImGui::Checkbox("2n", &nVis))
    {
        pN->setEnabled(nVis);
        nN->setEnabled(nVis);
    }
    if (ImGui::Checkbox("Polygon", &pVis))
    {
        polygon->setEnabled(pVis);
        //polygonP->setEnabled(pVis);
    }

    if (ImGui::SliderInt("Nx", &Nx, 0, 100))
    {
        diagonal = gridGernate(Nx, Ny, Nz);
        ImplicitValue(radius, h);
        marchingCubes(Nx, Ny, Nz, radius, h);
    }
    if (ImGui::SliderInt("Ny", &Ny, 0, 100))
    {
        diagonal = gridGernate(Nx, Ny, Nz);
        ImplicitValue(radius, h);
        marchingCubes(Nx, Ny, Nz, radius, h);
    }
    if (ImGui::SliderInt("Nz", &Nz, 0, 100))
    {
        diagonal = gridGernate(Nx, Ny, Nz);
        ImplicitValue(radius, h);
        marchingCubes(Nx, Ny, Nz, radius, h);
    }

    if (ImGui::SliderFloat("R", &radius, 0.0, 150.0))
    {
        ImplicitValue(radius, diagonal / 10.0);
        marchingCubes(Nx, Ny, Nz, radius, diagonal / 10.0);
    };
    if (ImGui::SliderFloat("R low_scale", &radius, 0.0, 0.5))
    {
        ImplicitValue(radius, diagonal / 10.0);
        marchingCubes(Nx, Ny, Nz, radius, diagonal / 10.0);
    };
    if (ImGui::SliderInt("Cube", &cube, 0, 100))
    {
        showCube(cube, Nx, Ny, Nz);
    }
    static int px=50, py=50;
    ImGui::SliderInt("px",&px,10,100);
    ImGui::SliderInt("py",&py,10,100);

    if(ImGui::Button("Ray tracing intersect")){
        rayTracing(px, py, radius, h);
    }
    static bool rt=false;  
    ImGui::Checkbox("show",&rt);
    if(rt){
        ImGui::Begin("My Tracing");
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
    
    ImVec2 p = ImGui::GetCursorScreenPos();
       for(int i = 0; i<px; i++){
            for(int j = 0; j<py; j++){
                ImVec2 p0 = ImVec2(p.x+(float)polyscope::view::windowWidth/2.0f/(float)px*float(i),p.y+(float)polyscope::view::windowHeight/2.0f/(float)py*float(j));
                ImVec2 p1 = ImVec2(p.x+(float)polyscope::view::windowWidth/2.0f/(float)px*float(i+1),p.y+(float)polyscope::view::windowHeight/2.0f/(float)py*float(j));
                ImVec2 p2 = ImVec2(p.x+(float)polyscope::view::windowWidth/2.0f/(float)px*float(i+1),p.y+(float)polyscope::view::windowHeight/2.0f/(float)py*float(j+1));
                ImVec2 p3 = ImVec2(p.x+(float)polyscope::view::windowWidth/2.0f/(float)px*float(i),p.y+(float)polyscope::view::windowHeight/2.0f/(float)py*float(j+1));
                if(contact[px*i+j])draw_list->AddQuadFilled(p0, p1, p2, p3, IM_COL32(255*(1.0-0.9*depth_Z[px*i+j]), 0, 0, 255));
                else draw_list->AddQuadFilled(p0, p1, p2, p3, IM_COL32(0, 0, 0, 255));
            }
        }
        

    
    ImGui::End();

    }*/
    /*RayTracing*/
      
    


    // Advance the ImGui cursor to claim space in the window (otherwise the window will appear small and needs to be resized)
    

    

    
    /*static bool extended = false;
    if (ImGui::Checkbox("extended marching cubes",&extended)){
        EmarchingCubes(Nx,Ny,Nz,radius,h);
    }*/


    /*static int m = 10;
    static int n = 10;
    ImGui::SliderInt("m",&m,0,100);
    ImGui::SliderInt("n",&n,0,100);
    static bool gridVis = false;
    ImGui::Checkbox("grid", &gridVis);
    if(pc != nullptr){
        grid.clear();
        float minX=10000.0, minY=10000.0, maxX=-10000.0, maxY=-10000.0;
        for(size_t i = 0; i<sds->getPoints().size(); i++){
            if(minX > sds->getPoints()[i][0]) minX = sds->getPoints()[i][0];
            if(minY > sds->getPoints()[i][1]) minY = sds->getPoints()[i][1];
            if(maxX < sds->getPoints()[i][0]) maxX = sds->getPoints()[i][0];
            if(maxY < sds->getPoints()[i][1]) maxY = sds->getPoints()[i][1];
        }
        float unitXL = (maxX-minX)/m , unitYL = (maxY-minY)/n;
        for(int i = 0; i <= m; i++){
            for(int j = 0; j <= n; j++){
                grid.push_back({minX + unitXL*i, minY + unitYL*j, 0.0});
            }
        }
        sds3 = std::make_unique<SpatialDataStructure>(grid);
        std::vector<std::array<int,4>> edges;
        for(int i = 0; i < m; i++){
            for(int j = 0; j< n; j++){
                edges.push_back({i*(n+1)+j, i*(n+1)+j+1});
                edges.push_back({i*(n+1)+j+1, i*(n+1)+j+n+2});
                edges.push_back({i*(n+1)+j+n+2, i*(n+1)+j+n+1});
                edges.push_back({i*(n+1)+j, i*(n+1)+j+n+1});
            }
        }

        grd = polyscope::registerCurveNetwork("grid",grid, edges);
        grd->setRadius(0.0005,true);
        grd->setEnabled(gridVis);
    }

    static float R=0.2;

    static bool WLSvis = false;
    ImGui::Checkbox("WLS",&WLSvis);
    if(ImGui::SliderFloat("Radius",&R,0.1,1.0)){
        WLS.clear();
        struct WeightCalculation{
            static float weight(Point fixedPoint, Point X) {
                float d = EuclideanDistance::measure(fixedPoint,X);
                if(d<R) return std::pow(1-d,4)*(4*d+1);
                else return 0;
            }
        };
        PointList a;
        for(size_t i = 0 ; i < sds->getPoints().size() ; i++){

            a.push_back(Point {sds->getPoints()[i][0],sds->getPoints()[i][1],0.0});
        }
        sds2 = std::make_unique<SpatialDataStructure>(a);
        for(size_t i = 0 ; i < sds3->getPoints().size(); i++){
            Eigen::Matrix<float, 6, 6> ft; ft.setZero();
            Eigen::Matrix<float, 6, 1> st; st.setZero();
            Point fixed = sds3->getPoints()[i];
            std::vector<std::size_t> InRad = sds2->collectInRadius(fixed,R);
            for(size_t j = 0 ; j < InRad.size() ; j++){
                Point training = sds->getPoints()[InRad[j]];
                float W = WeightCalculation::weight(grid[i], Point {training[0], training[1], 0.0});
                Eigen::Matrix<float, 6, 1> bx {1, training[0] , training[1], training[0]*training[0], training[0]*training[1], training[1]*training[1]};
                ft += W*bx*bx.transpose(), st += W*bx*training[2];
            }
            Eigen::Matrix<float,6,1> c = ft.inverse()*st;
            float x = fixed[0], y = fixed[1];
            Eigen::Matrix<float,1,6> Basis {1,x,y,x*x,x*y,y*y};
            float z = Basis*c;
            WLS.push_back({x,y,z});
        }
        wlsp=polyscope::registerPointCloud("WLS",WLS);

        std::vector<std::array<int,2>> WLSedges;
        for(int i = 0; i < m; i++){
            for(int j = 0; j< n; j++){
                WLSedges.push_back({i*(n+1)+j, i*(n+1)+j+1});
                WLSedges.push_back({i*(n+1)+j+1, i*(n+1)+j+n+2});
                WLSedges.push_back({i*(n+1)+j+n+2, i*(n+1)+j+n+1});
                WLSedges.push_back({i*(n+1)+j, i*(n+1)+j+n+1});
            }
        }
        wls = polyscope::registerCurveNetwork("WLSnet", WLS, WLSedges);
        wls->setRadius(0.001,true);
        wls->setEnabled(WLSvis);
        wlsp->setEnabled(WLSvis);

    }*/

    /*static int kk = 1.0;
    ImGui::SliderInt("kk",&kk,1,10);
    static bool BezierSurface = false;
    static bool BezierSurfaceNormal = false;
    ImGui::Checkbox("BezierSurface",&BezierSurface);
    ImGui::Checkbox("BezierSurfaceNormal",&BezierSurfaceNormal);

    if(wls != nullptr){
        PointList BSpoints;
        std::vector<std::array<int,3>> BSedges;
        PointList NormalPoints;
        std::vector<std::array<int,2>> NormalEdges;
        for(int i = 0; i <= m*kk; i++){
            for(int j = 0; j <= n*kk; j++){
                Point point = Bezier::q(m,n,(float) 1.0/m/kk*i,(float) 1.0/n/kk*j,WLS);
                BSpoints.push_back(point);
                NormalPoints.push_back(point);
                std::array<float,3> direction = Bezier::normal(m, n,(float) 1.0/m/kk*i,(float) 1.0/n/kk*j, WLS);
                Point normal {point[0]+direction[0], point[1]+direction[1], point[2]+direction[2] };
                NormalPoints.push_back(normal);
            }
        }
        for(int i = 0; i < m*kk; i++){
            for(int j = 0; j< n*kk; j++){
                BSedges.push_back({i*(n*kk+1)+j, i*(n*kk+1)+j+1, i*(n*kk+1)+j+n*kk+1});
                BSedges.push_back({i*(n*kk+1)+j+1, i*(n*kk+1)+j+n*kk+2, i*(n*kk+1)+j+n*kk+1});
            }
        }
        for(int i = 0; i < (int) NormalPoints.size()/2 ; i++){
            NormalEdges.push_back({2*i, 2*i + 1});
        }
        bsf = polyscope::registerSurfaceMesh("BezierSurface",BSpoints, BSedges);
        nml = polyscope::registerCurveNetwork("BezierNormal",NormalPoints, NormalEdges);
        nml->setRadius(0.002, true);
        bsf->setEnabled(BezierSurface);
        nml->setEnabled(BezierSurfaceNormal);
    }
    if(ImGui::Button("MLS")){
        PointList MLSpoints;
        std::vector<std::array<int,3>> MLSedges;
        struct WeightCalculation{
            static float weight(Point fixedPoint, Point X) {
                float d = EuclideanDistance::measure(fixedPoint,X);
                if(d<R) return std::pow(1-d,4)*(4*d+1);
                else return 0;
            }
            static Point mls(Point X){
                Eigen::Matrix<float, 6, 6> ft; ft.setZero();
                Eigen::Matrix<float, 6, 1> st; st.setZero();
                for(size_t j = 0 ; j < sds->getPoints().size() ; j++){
                    Point training = sds->getPoints()[j];
                    float W = WeightCalculation::weight(X, Point {training[0], training[1], 0.0});
                    if(W == 0.0) continue;
                    Eigen::Matrix<float, 6, 1> bx {1, training[0] , training[1], training[0]*training[0], training[0]*training[1], training[1]*training[1]};
                    ft += W*bx*bx.transpose(), st += W*bx*training[2];
                }
                Eigen::Matrix<float,6,1> c = ft.inverse()*st;
                float x = X[0], y = X[1];
                Eigen::Matrix<float,1,6> Basis {1,x,y,x*x,x*y,y*y};
                float z = Basis*c;
                Point point {x,y,z};
                return point;
            }
        };
        for(int i = 0; i <= kk*m; i++){
            for(int j = 0; j <= kk*n ; j++){
                Point input {(float) 1.0/kk/m*i, (float) 1.0/kk/n*j, 0.0};
                MLSpoints.push_back(WeightCalculation::mls(input));
            }
        }
        for(int i = 0; i < m*kk; i++){
            for(int j = 0; j< n*kk; j++){
                MLSedges.push_back({i*(n*kk+1)+j, i*(n*kk+1)+j+1, i*(n*kk+1)+j+n*kk+1});
                MLSedges.push_back({i*(n*kk+1)+j+1, i*(n*kk+1)+j+n*kk+2, i*(n*kk+1)+j+n*kk+1});
            }
        }

        polyscope::registerSurfaceMesh("MLS",MLSpoints,MLSedges);
    }
    if(ImGui::Button("mess")){
    if(polyscope::pick::getSelection().first != nullptr){
            std::vector<size_t> InR= sds->collectInRadius(sds3->getPoints()[30], R);
            PointList aa;
            if(!InR.empty()) polyscope::warning("오 ㅅㅂ 됐다.");
            for(size_t i = 0; i<InR.size();i++){
                Point b = sds2->getPoints()[InR[i]];
                aa.push_back(b);
            }
            polyscope::registerPointCloud("shit",aa);
        }
    }


    // TODO: Implement radius search
    // TODO: Implement visualizations
    static const char* mode[]={"radius", "knn"};
    static int current = 0;
    static float radius = 0.100;
    static int k = 10;
    //static size_t n = pc->nPoints();
    ImGui::Text("Query");
    ImGui::Combo("Mode", &current, mode, IM_ARRAYSIZE(mode));
    ImGui::SliderFloat("radius", &radius, 0.000, 1.000);
    ImGui::SliderInt("k", &k, 0, 10000);
    if(current == 0){
        if(polyscope::pick::getSelection().first == nullptr) return;
        std::size_t idx = polyscope::pick::getSelection().second;
        std::vector<std::size_t> pointsId = sds->collectInRadius(sds->getPoints()[idx],radius);
        std::vector<std::array<double,3>> colorSet;
        float r = pc->getPointColor()[0];
        float g = pc->getPointColor()[1];
        float b = pc->getPointColor()[2];
        std::array<double,3> hl = {{0.8, 0.8, 0.0}};
        for(size_t i = 0; i<sds->getPoints().size(); i++){
            colorSet.push_back({{r,g,b}});
        }
        for(size_t i = 0; i<pointsId.size(); i++){
            colorSet.erase(colorSet.begin() + pointsId[i]);
            colorSet.insert(colorSet.begin() + pointsId[i], hl);
        }

        pc->addColorQuantity("color",colorSet);
    }
    else{
        if(polyscope::pick::getSelection().first == nullptr) return;
        std::size_t idx = polyscope::pick::getSelection().second;
        std::vector<std::size_t> pointsId = sds->collectKNearest(sds->getPoints()[idx],k);
        std::vector<std::array<double,3>> colorSet;
        float r = pc->getPointColor()[0];
        float g = pc->getPointColor()[1];
        float b = pc->getPointColor()[2];
        std::array<double,3> hl = {{0.8, 0.8, 0.0}};
        for(size_t i = 0; i<sds->getPoints().size(); i++){
            colorSet.push_back({{r,g,b}});
        }
        for(size_t i = 0; i<pointsId.size(); i++){
            colorSet.erase(colorSet.begin() + pointsId[i]);
            colorSet.insert(colorSet.begin() + pointsId[i], hl);
        }

        pc->addColorQuantity("color",colorSet);
    }*/
}

int main(int argc, char **argv)
{
    // Configure the argument parser
    args::ArgumentParser parser("Computer Graphics 2 Sample Code.");

    // Parse args
    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Help &)
    {
        std::cout << parser;
        return 0;
    }
    catch (const args::ParseError &e)
    {
        std::cerr << e.what() << std::endl;

        std::cerr << parser;
        return 1;
    }

    // Options
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
    polyscope::options::shadowBlurIters = 6;

    // Initialize polyscope
    polyscope::init();

    // Add a few gui elements
    polyscope::state::userCallback = callback;

    
 
    // Show the gui
    polyscope::show();

    return 0;
}
