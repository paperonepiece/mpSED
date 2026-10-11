import os
import sys
import math
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
import numpy as np
from fractions import Fraction

import Parsers

def error(string):
    print('\nERROR: {}!\n'.format(string),flush=True)
    exit()
    
class lattice:
    """
    Lattice and reciprocal-space q-point manager.

    Coordinate conventions
    ----------------------
    Direct primitive vectors are stored by row:

        dir_lat[0, :] = a1
        dir_lat[1, :] = a2
        dir_lat[2, :] = a3

    Reciprocal primitive vectors are also stored by row:

        recip_vecs[0, :] = b1
        recip_vecs[1, :] = b2
        recip_vecs[2, :] = b3

    Therefore:

        a_i dot b_j = 2*pi*delta_ij

    If q_reduced is represented as a row vector:

        q_reduced = [h, k, l]

    then:

        q_cart = q_reduced @ recip_vecs

    This is equivalent to:

        q_cart_column = recip_vecs.T @ q_reduced_column
    """
    def __init__(self,
                 params,
                 eigen_vectors):

        # read the unitcell and basis positions from a file
        self.atom_ids, self.unit_cells, self.basis_pos, self.masses = (Parsers.
                parse_lattice_file(params))
        self.cell_ref_ids = np.argwhere(self.basis_pos == 1)
        self.cell_ref_ids = self.cell_ref_ids.reshape(len(self.cell_ref_ids))

        self.q_fraction_max_denominator = int(
            getattr(
                params,
                "q_fraction_max_denominator",
                48,
            )
        )

        self.q_fraction_tolerance = float(
            getattr(
                params,
                "q_fraction_tolerance",
                1.0e-3,
            )
        )

        if self.q_fraction_max_denominator < 1:
            error("q_fraction_max_denominator must be positive")
        if self.q_fraction_tolerance < 0:
            error("q_fraction_tolerance must be non-negative")
        
        self.q_zero_tolerance = float(getattr(params, "q_zero_tolerance", 1.0e-12))
        
        self.supercell_matrix = self._normalize_supercell_matrix(
                    getattr(params, "supercell_dim", None)
        )
        # print('supercell matrix: ',self.supercell_matrix)
        # calculate direct lattice vectors
        self.dir_lat = np.zeros((3, 3))
        prim_vecs = params.prim_vecs.copy()

        self.dir_lat[0,:] = prim_vecs[0,:]*params.lat_params[0]
        self.dir_lat[1,:] = prim_vecs[1,:]*params.lat_params[1]
        self.dir_lat[2,:] = prim_vecs[2,:]*params.lat_params[2]
        
        # calculate reciprocal lattice vectors
        self.recip_vecs = np.zeros((3,3))
        self.bz_vol = self.dir_lat[0,:].dot(np.cross(self.dir_lat[1,:], self.dir_lat[2,:]))
        self.recip_vecs[0,:] = 2*np.pi*np.cross(self.dir_lat[1,:], self.dir_lat[2,:])/self.bz_vol
        self.recip_vecs[1,:] = 2*np.pi*np.cross(self.dir_lat[2,:], self.dir_lat[0,:])/self.bz_vol
        self.recip_vecs[2,:] = 2*np.pi*np.cross(self.dir_lat[0,:], self.dir_lat[1,:])/self.bz_vol
        
        # test if reciprocal lattice vectors converted correctly
        if abs(self.bz_vol) < 1e-14:
            raise ValueError(
                "\nERROR: Primitive cell volume is zero\n"
            )
            
        expected = (np.sign(self.bz_vol) * 2 * np.pi * np.eye(3))
        actual = self.dir_lat @ self.recip_vecs.T
        # print('dir-recip conversion matrix:',self.dir_lat @ self.recip_vecs.T)
        if not np.allclose(actual, expected, rtol=1e-10, atol=1e-10):
            error_value = np.max(np.abs(actual - expected))
            error("Direct and reciprocal lattice vectors are inconsistent: "
                  "max error = {}".format(error_value))
            
        if params.with_eigs: # use phonopy q-points
            print('\nCOMMENT: Getting q-points from PHONOPY \'{}\' file\n'
                    .format(params.eigvecs_file),flush=True)
            params.num_qpoints = list(map(int,[eigen_vectors.num_qpoints]))
            self.num_qpoints = list(map(int,[eigen_vectors.num_qpoints]))
            # phonopy provides reduced q-points
            self.reduced_qpoints = eigen_vectors.qpoints
            # self.qpoints = np.copy(self.reduced_qpoints)
            # convert to recip. lattice-vector units
            self.qpoints = self._reduced_to_cartesian(self.reduced_qpoints)
            # for i in range(sum(self.num_qpoints)):
                
            #     self.qpoints[i,:] = self.recip_vecs.T.dot(self.qpoints[i,:])

        else: # create q-point list from input file
            # construct the BZ paths
            # self.construct_BZ_path(params)
            self.construct_BZ_path_auto(params)
            # (params)
                
        self._print_qpoints()    
        
    def _check_q_periodicity_reduced(
        self,
        reduced_qpoints,
        supercell_matrix,
        tolerance=1.0e-10,
    ):
        reduced_qpoints = np.asarray(
            reduced_qpoints,
            dtype=float,
        )

        if reduced_qpoints.ndim == 1:
            reduced_qpoints = reduced_qpoints.reshape(1, 3)

        transformed = (
            reduced_qpoints @ supercell_matrix.T
        )

        error = np.abs(
            transformed - np.rint(transformed)
        )

        print("q * S.T:")
        print(transformed)

        print("distance to integer:")
        print(error)

        if np.max(error) > tolerance:
            raise ValueError(
                "q-points are not commensurate with the supercell"
            )

    def _check_q_periodicity(self, qpoints, tolerance=1.0e-10):
        """
        Check q-point periodicity against the supercell.

        qpoints are Cartesian reciprocal coordinates with shape (3,)
        or (N, 3). The supercell vectors are constructed using the
        convention A_super = S @ A_primitive.
        """
        qpoints = np.asarray(qpoints, dtype=float)

        single_point = qpoints.ndim == 1
        if single_point:
            qpoints = qpoints.reshape(1, 3)

        if qpoints.ndim != 2 or qpoints.shape[1] != 3:
            raise ValueError(
                "qpoints must have shape (3,) or (N, 3), got {}".format(
                    qpoints.shape
                )
            )

        supercell_vectors = (
            self.supercell_matrix @ self.dir_lat
        )

        # Each row is a supercell translation vector. For a compatible q,
        # q dot R is an integer multiple of 2*pi.
        phases = qpoints @ supercell_vectors.T
        phase_factors = np.exp(1j * phases)

        errors = np.abs(phase_factors - 1.0)
        max_error = float(errors.max()) if errors.size else 0.0

        if max_error > tolerance:
            bad_point = int(
                np.unravel_index(np.argmax(errors), errors.shape)[0]
            )
            raise ValueError(
                "q-point is not periodic on the simulation supercell: "
                "point index {}, q={}, max phase error={:.3e}".format(
                    bad_point,
                    qpoints[bad_point],
                    max_error,
                )
            )

        if single_point:
            return phase_factors[0]

        return phase_factors

    
    def _print_qpoints(self):
        print("\nThere are {} q-points to be calculated:\n".format(
            len(self.reduced_qpoints)),
            flush=True
        )

        # for index, qpoint in enumerate(
        #     self.reduced_qpoints
        # ):
        #     print(
        #         "\tq-point {:d}: "
        #         "(reduced) q=({:.6f}, {:.6f}, {:.6f}) "
        #         "(cartesian) q=({:.6f}, {:.6f}, {:.6f})".format(
        #             index + 1,
        #             qpoint[0],
        #             qpoint[1],
        #             qpoint[2],
        #             self.qpoints[index, 0],
        #             self.qpoints[index, 1],
        #             self.qpoints[index, 2]
        #         ),
        #         flush=True
        #     )
            
        for index, qpoint in enumerate(self.reduced_qpoints):    
            print("\tq-point {:2d}: "
                  "(reduced) q=({:.4f}, {:.4f}, {:.4f}) ".format(
                        index + 1,
                        qpoint[0],
                        qpoint[1],
                        qpoint[2]),
                flush=True
            )
        print(flush=True)
    
    def _normalize_supercell_matrix(self, supercell_dim):
        """
        Normalize either of the following input formats:

            [Nx, Ny, Nz]

        or:

            [[s11, s12, s13],
             [s21, s22, s23],
             [s31, s32, s33]]

        The returned matrix is an integer 3x3 matrix S satisfying:

            A_super = S @ A_primitive
        """

        if supercell_dim is None:
            error("supercell_dim is required")

        raw_sdim = np.asarray(supercell_dim)

        # Backward-compatible diagonal form:
        #
        #     SUPERCELL_DIM = Nx Ny Nz
        #
        if raw_sdim.shape == (3,):
            values = []

            for value in raw_sdim:
                if not np.isfinite(float(value)):
                    error("supercell dimensions must be finite")

                integer_value = int(value)

                if float(value) != integer_value:
                    error("supercell dimensions must be integers")

                values.append(integer_value)

            new_smatrix = np.diag(values).astype(np.int64)

        elif raw_sdim.shape == (3, 3):
            new_smatrix = np.zeros(
                (3, 3),
                dtype=np.int64,
            )

            for i in range(3):
                for j in range(3):
                    value = raw_sdim[i, j]

                    if not np.isfinite(float(value)):
                        error("supercell_matrix must contain finite values")

                    integer_value = int(value)

                    if float(value) != integer_value:
                        error("supercell_matrix must contain integers")

                    new_smatrix[i, j] = integer_value

        else:
            error("supercell_dim must have shape (3,) or (3, 3), "
                  "got {}".format(raw_sdim.shape))

        determinant = round(
            np.linalg.det(new_smatrix)
        )

        if determinant == 0:
            error("supercell_matrix must be nonsingular")

        return new_smatrix
    
    def _fraction_from_value(self, value):
            """
            Convert a q coordinate into a controlled Fraction.
    
            Exact fraction strings are preserved:
    
                "1/3" -> Fraction(1, 3)
    
            Decimal strings and floats are first interpreted as their
            decimal representation:
    
                "0.333" -> Fraction(333, 1000)
    
            Then a low-denominator candidate is considered:
    
                Fraction(333, 1000) -> Fraction(1, 3)
    
            only if the difference is within q_fraction_tolerance.
    
            Therefore:
    
                0.333 -> 1/3
    
            with the default tolerance, but arbitrary high-denominator
            fractions are not silently introduced.
            """
            if isinstance(value, Fraction):
                return value
    
            if isinstance(value, str):
                text = value.strip()
    
                if not text:
                    error("Empty q-point coordinate")
    
                # Preserve explicit fractions exactly.
                if "/" in text:
                    try:
                        return Fraction(text)
                    except ValueError:
                        error("Invalid rational q coordinate: {!r}".format(value))   
                try:
                    numeric_value = float(text)
                except ValueError:
                    error("Invalid q coordinate: {!r}".format(value))
            else:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Invalid q coordinate: {!r}".format(
                            value
                        )
                    ) from exc
    
            if not np.isfinite(numeric_value):
                error("q coordinate must be finite: {!r}".format(value))
    
            # The string representation avoids using the full binary
            # floating-point expansion.
            decimal_fraction = Fraction(str(numeric_value))
    
            candidate = decimal_fraction.limit_denominator(
                self.q_fraction_max_denominator
            )
    
            candidate_error = abs(
                float(candidate) - numeric_value
            )
    
            if candidate_error <= self.q_fraction_tolerance:
                return candidate
    
            # Keep the supplied decimal value rather than guessing.
            return decimal_fraction 
    
    def _fraction_vector(self, values):
        """Convert a three-component vector to Fractions."""

        if len(values) != 3:
            raise ValueError(
                "q-point must contain exactly three coordinates"
            )

        return tuple(
            self._fraction_from_value(value)
            for value in values
        )
    
    def _remove_signed_zero(self, qpoints):
        """
        Convert numerically negligible values to +0.0.

        This removes values such as -1.5e-16 caused by floating-point
        roundoff, while preserving genuine negative components.
        """
        qpoints = np.asarray(
            qpoints,
            dtype=float,
        ).copy()

        mask = np.abs(qpoints) < self.q_zero_tolerance
        qpoints[mask] = 0.0

        return qpoints

    def _reduced_to_cartesian(self, reduced_qpoints):
        """
        Convert an array of reduced q-points to Cartesian q-points.

        Input:
            shape (N, 3)

        Output:
            shape (N, 3)
        """

        reduced_qpoints = np.asarray(
            reduced_qpoints,
            dtype=float,
        )

        if reduced_qpoints.ndim != 2:
            raise ValueError(
                "reduced_qpoints must be two-dimensional"
            )

        if reduced_qpoints.shape[1] != 3:
            raise ValueError(
                "reduced_qpoints must have shape (N, 3)"
            )
            
        cartesian_qpoints = reduced_qpoints @ self.recip_vecs 
        
        return self._remove_signed_zero(cartesian_qpoints)
           
    
    def _transformed_reduced_q(self, q_reduced, supercell_matrix=None):
        """
        Calculate:

            u = S.T @ q_reduced

        using exact Fraction arithmetic.

        A q point is commensurate iff every component of u is integer.
        """

        if supercell_matrix is None:
            supercell_matrix = self.supercell_matrix

        S = self._normalize_supercell_matrix(
            supercell_matrix
        )

        q = self._fraction_vector(
            q_reduced
        )

        transformed = []

        for i in range(3):
            value = sum(
                Fraction(int(S[j, i])) * q[j]
                for j in range(3)
            )

            transformed.append(value)

        return tuple(transformed)
    
    def is_commensurate(
        self,
        q_reduced,
        supercell_matrix=None,
    ):
        """
        Test a reduced q point using:

            S.T @ q_reduced in Z^3
        """

        transformed = self._transformed_reduced_q(
            q_reduced,
            supercell_matrix,
        )

        return all(
            value.denominator == 1
            for value in transformed
        )
    
    def _ceil_fraction(self, value):
        return -((-value.numerator) // value.denominator)

    def _floor_fraction(self, value):
        return value.numerator // value.denominator
    
    def _fraction_from_float(self, value):
            '''
            str() avoids binary floating-point artifacts such as
            0.020000000000000004.
            '''
            return Fraction(str(float(value))).limit_denominator(1000000)
        
    def _commensurate_segment(self,
                              q_start,
                              q_end,
                              supercell_matrix=None):
        """
        Return all commensurate points on a straight reduced-q segment.

        q(t) = q_start + t * (q_end - q_start)
        0 <= t <= 1

        For:

            A_super = S @ A_primitive

        the commensurability condition is:

            S.T @ q(t) = integer vector
        """

        if supercell_matrix is None:
            supercell_matrix = self.supercell_matrix

        Sc = self._normalize_supercell_matrix(supercell_matrix )

        start = self._fraction_vector(q_start)
        end = self._fraction_vector(q_end)

        # Transform the path endpoints:
        #
        #     u(t) = S.T @ q(t)
        #
        # Every component of u(t) must be integer.
        transformed_start = []
        transformed_end = []

        for i in range(3):
            start_value = sum(
                Fraction(int(Sc[j, i])) * start[j]
                for j in range(3)
            )

            end_value = sum(
                Fraction(int(Sc[j, i])) * end[j]
                for j in range(3)
            )

            transformed_start.append(start_value)
            transformed_end.append(end_value)

        allowed_t = None

        for a, b in zip(transformed_start,
                        transformed_end):
            
            slope = b - a

            # The transformed coordinate is constant along this path.
            if slope == 0:
                if a.denominator != 1:
                    return []
                continue

            low = min(a, b)
            high = max(a, b)

            first_integer = self._ceil_fraction(low)

            last_integer = self._floor_fraction(high)

            component_t = set()

            for integer_value in range(first_integer,
                                       last_integer + 1):
                
                t = (Fraction(integer_value) - a) / slope

                if Fraction(0) <= t <= Fraction(1):
                    component_t.add(t)

            if allowed_t is None:
                allowed_t = component_t
            else:
                allowed_t.intersection_update(component_t)

            if not allowed_t:
                return []

        # Entire path is constant in transformed coordinates.
        if allowed_t is None:
            allowed_t = {Fraction(0), Fraction(1)}

        points = []

        for t in sorted(allowed_t):
            point = tuple(start[axis] + t * (end[axis] - start[axis]) for axis in range(3))
            points.append(point)

        return points
    
    def construct_BZ_path_auto(self, params):
        """
        Construct all supercell-commensurate q-points along the requested
        symmetry path.

        params.qsym_points:
            Reduced reciprocal coordinates.

        Example:
            [
                [0,   0,   0],
                [0.5, 0.5, 0],
            ]

        params.num_qpaths:
            Number of path segments.

        params.num_qpoints:
            Requested number of q-points per segment.

        The actual number of points is determined by supercell
        commensurability. The requested NUM_QPOINTS value is therefore
        treated as a request and may be overridden.
        """

        qsym_points = np.asarray(
            params.qsym_points,
            dtype=object,
        )

        if qsym_points.ndim == 1:
            if len(qsym_points) % 3 != 0:
                error("QSYM_POINTS must contain groups of three coordinates")

            qsym_points = qsym_points.reshape(
                (-1, 3)
            )

        if qsym_points.ndim != 2:
            error("QSYM_POINTS must be a two-dimensional array")
            

        if qsym_points.shape[1] != 3:
            error("QSYM_POINTS must have shape (N, 3), got {}".format(
                                qsym_points.shape)
            )

        number_of_paths = int(params.num_qpaths)

        if qsym_points.shape[0] != number_of_paths + 1:
            error("QSYM_POINTS must contain NUM_QPATHS + 1 points: "
                            "expected {}, got {}".format(
                                number_of_paths + 1,
                                qsym_points.shape[0],
                            )
            )

        requested_num_qpoints = list(
            map(int, np.asarray(params.num_qpoints).reshape(-1))
        )

        if len(requested_num_qpoints) != number_of_paths:
            error("NUM_QPOINTS must contain one value per q-path: "
                            "expected {}, got {}".format(
                                number_of_paths,
                                len(requested_num_qpoints),
                            )
            )
            
        detected_num_qpoints = []
        reduced_segments = []

        for path_index in range(number_of_paths):
            q_start = qsym_points[path_index]
            q_end = qsym_points[path_index + 1]

            segment = self._commensurate_segment(
                q_start,
                q_end,
                self.supercell_matrix,
            )

            if len(segment) == 0:
                raise ValueError(
                    "No commensurate q-points found for q-path {}: "
                    "{} -> {}".format(
                        path_index + 1,
                        q_start,
                        q_end,
                    )
                )

            # Do not duplicate the endpoint at the start of the next
            # segment. The final point of the entire path is also omitted
            # with this endpoint=False convention.
            segment_without_endpoint = segment[:-1]

            # If the segment only contains its endpoint, keep its first
            # point so that the segment does not become empty.
            if len(segment_without_endpoint) == 0:
                segment_without_endpoint = segment[:1]

            reduced_segments.append(segment_without_endpoint)
            number_detected = len(segment_without_endpoint)

            detected_num_qpoints.append(number_detected)

            number_requested = requested_num_qpoints[path_index]

            if number_requested != number_detected:
                print(
                    "\nCOMMENT: Q-point number overridden for "
                    "path segment {}: {} -> {}, user-requested {} "
                    "but using {} commensurate points instead".format(
                        path_index + 1,
                        q_start,
                        q_end,
                        number_requested,
                        number_detected
                    ),
                    flush=True
                )

        if reduced_segments:
            reduced_fractional_points = []

            for segment in reduced_segments:
                reduced_fractional_points.extend(segment)

            # Convert exact Fraction points to floats only after all
            # commensurability calculations are complete.
            self.reduced_qpoints = np.asarray(
                [
                    [float(value) for value in point]
                    for point in reduced_fractional_points
                ],
                dtype=float,
            )

        else:
            self.reduced_qpoints = np.empty(
                (0, 3),
                dtype=float,
            )

        self.num_qpoints = detected_num_qpoints

        # Keep params synchronized with the actual path.
        params.num_qpoints = list(
            detected_num_qpoints
        )

        self.qpoints = self._reduced_to_cartesian(self.reduced_qpoints)
    
    def _validate_supercell_matrix(supercell_matrix):
            """Return a validated, nonsingular 3x3 integer supercell matrix."""
            raw_smatrix = np.asarray(supercell_matrix)
    
            if raw_smatrix.shape != (3, 3):
                raise ValueError(
                    "supercell_matrix must have shape (3, 3), got {}".format(raw_smatrix.shape)
                )
    
            try:
                new_smatrix = np.asarray(
                        [[int(value) for value in row] for row in raw_smatrix],
                        dtype=np.int64,
                )
            except (TypeError, ValueError, OverflowError):
                raise ValueError("supercell_matrix must contain integers")
    
            # Reject values such as 1.5 rather than silently truncating them.
            for i in range(3):
                for j in range(3):
                    if raw_smatrix[i, j] != new_smatrix[i, j]:
                        raise ValueError("supercell_matrix must contain integers")
    
            determinant = round(np.linalg.det(new_smatrix))
            if determinant == 0:
                raise ValueError("supercell_matrix must be nonsingular")
    
            return new_smatrix
    
    def _commensurate_segment1(self,
                                  q_start,
                                  q_end,
                                  supercell_dim):
                """
                Return all commensurate points on a straight reduced-q path.
        
                q_start and q_end are reduced reciprocal coordinates (h,k,l):
                    q = h*b1 + k*b2 + l*b3
        
                A point is commensurate with the supercell when:
                    q[i] * supercell_dim[i] is an integer
                """
                        
                if len(q_start) != 3 or len(q_end) != 3:
                    raise ValueError('Q-path endpoints must be three-dimensional, not {} -> {}'.format(q_start, q_end))
                
                # Each coordinate produces a set of allowed path parameters t.
                # q(t) = q_start + t*(q_end-q_start), 0 <= t <= 1.
                allowed_t = None
                has_varying_coordinate = False
                self.q_fraction_max_denominator = 48
                self.q_fraction_tolerance = 1.0e-3
                
                for axis in range(3):
                    ncell = int(supercell_dim[axis])
        
                    a = self._fraction_from_float(q_start[axis]) * ncell
                    b = self._fraction_from_float(q_end[axis]) * ncell
                    
                    slope = b - a
        
                    if slope == 0:
                        # A constant coordinate must itself be commensurate.
                        if a.denominator != 1:
                            return []
                        # axis_t = {Fraction(0), Fraction(1)}
                        continue
                    
                    has_varying_coordinate = True
                    low = min(a, b)
                    high = max(a, b)
                    
                    first_integer = self._ceil_fraction(low) # 0
                    last_integer = self._floor_fraction(high) # 25
                    
                    axis_t = set()
    
                    for integer_value in range(first_integer, 
                                               last_integer + 1
                                               ):
                        t = (Fraction(integer_value) - a) / slope
    
                        if Fraction(0) <= t <= Fraction(1):
                            axis_t.add(t)
                        
                    if allowed_t is None:
                        allowed_t = axis_t
                    else:
                        allowed_t &= axis_t
        
                    if not allowed_t:
                        return []
                    
                if not has_varying_coordinate:
                    print('\nCAUTION: Constant Path for {} -> {},'
                          ' no q-point variation\n'.format(q_start,
                                                           q_end))
                    return q_start.reshape(1, 3)
                
                t_values = sorted(allowed_t)
                
                return np.array(
                    [
                        q_start + float(t) * (q_end - q_start)
                        for t in t_values
                    ],
                    dtype=float
                )   
    
    def construct_BZ_path_auto1(self, params):
        """
        Construct Q points using the requested symmetry path.

        params.qsym_points contains reduced reciprocal coordinates.
        params.supercell_dim contains the number of conventional cells
        along each real-space direction.
        """
        qsym_points = np.asarray(params.qsym_points, dtype=float)
        supercell_dim = np.asarray(params.supercell_dim, dtype=int)

        if qsym_points.ndim != 2 or qsym_points.shape[1] != 3:
            # print(qsym_points, qsym_points.ndim, qsym_points, qsym_points.shape)
            error('QSYM_POINTS must contain three-dimensional points')

        if qsym_points.shape[0] != params.num_qpaths+1:
            error('QSYM_POINTS must contain points = NUM_QPOINTS+1')
        
        supercell_dim = self._validate_supercell_matrix(supercell_dim)
        
        requested_num_qpoints:list[int] = list(map(int, params.num_qpoints))
        requested_num_qaths:int  = params.num_qpaths
        detected_num_qpoints:list[int] = []
        reduced_segments:list[np.ndarray] = []

        for path_index in range(requested_num_qaths):
            q_start = qsym_points[path_index]
            q_end = qsym_points[path_index + 1]
            
            segment = self._commensurate_segment1(
                q_start,
                q_end,
                supercell_dim
            )

            if len(segment) == 0:
                error(
                    'No commensurate Q points found for Q-path {}: '
                    '{} -> {}'.format(path_index + 1, 
                                      q_start, 
                                      q_end)
                )

            # Preserve the existing endpoint=False convention:
            # the endpoint of one segment is the start of the next segment.
            segment_without_endpoint = segment[:-1]

            # A segment containing only its two endpoints contributes one
            # point under the existing endpoint=False convention.
            if len(segment_without_endpoint) == 0:
                segment_without_endpoint = segment[:1]

            reduced_segments.append(segment_without_endpoint)
            detected_num_qpoints.append(len(segment_without_endpoint))

            num_requested = requested_num_qpoints[path_index]
            num_detected = detected_num_qpoints[-1]

            if num_requested != num_detected:
                print(
                    '\nCOMMENT: Q-point number overridden for path segment {}: '
                    '{} -> {}, user-requested {} but using the detected '
                    '{} commensurate points instead'.format(
                        path_index + 1,
                        q_start,
                        q_end,
                        num_requested,
                        num_detected
                    ),
                    flush=True
                )

        if reduced_segments:
            self.reduced_qpoints = np.vstack(reduced_segments)
        else:
            self.reduced_qpoints = np.empty((0, 3), dtype=float)

        # Store the actual number of points. All downstream code should use
        # lattice.num_qpoints after this function returns.
        self.num_qpoints = detected_num_qpoints

        # Convert reduced coordinates to Cartesian reciprocal coordinates.
        self.qpoints = np.zeros_like(self.reduced_qpoints)
        
        for i in range(len(self.reduced_qpoints)):
            # self.qpoints[i, :] = self.recip_vecs.dot(self.reduced_qpoints[i, :])
            self.qpoints[i, :] = self.recip_vecs.T.dot(self.reduced_qpoints[i, :])
        # self.qpoints = self.reduced_qpoints @ self.recip_vecs    
  
        # print(
        #     '\nThere are {} commensurate q-points to be calculated:\n'.format(
        #         len(self.qpoints)
        #     ),
        #     flush=True
        # )

        # for i in range(len(self.qpoints)):
        #     print(
        #         '\t(reduced) q=({:.6f}, {:.6f}, {:.6f})'.format(
        #             self.reduced_qpoints[i, 0],
        #             self.reduced_qpoints[i, 1],
        #             self.reduced_qpoints[i, 2]
        #         ),
        #         flush=True
        #     )
            
    def construct_BZ_path(self, params):
        # convert reduced q-points to recip. lattice-vector units
        self.qsym_points = np.copy(params.qsym_points)
        for i in range(len(params.qsym_points[:,0])):
            # self.qsym_points[i,:] = self.recip_vecs.dot(params.qsym_points[i,:])
            self.qsym_points[i,:] = self.recip_vecs.T.dot(params.qsym_points[i,:])
        self.num_qpoints = params.num_qpoints
        self.qpoints = np.zeros((sum(self.num_qpoints),3)) 
        self.reduced_qpoints = np.zeros((sum(self.num_qpoints),3))

        # make q-points list by linear interpolation between symmetry points
        start = 0
        for i in range(len(self.num_qpoints)):
            end = start+self.num_qpoints[i]    
            for j in range(3):
                self.qpoints[start:end,j] = np.linspace(self.qsym_points[i,j],
                                                        self.qsym_points[(i+1),j],
                                                        self.num_qpoints[i],
                                                        endpoint=False)
                self.reduced_qpoints[start:end,j] = np.linspace(params.qsym_points[i,j],
                                                                params.qsym_points[(i+1),j],
                                                                self.num_qpoints[i],
                                                                endpoint=False)
            start = start + self.num_qpoints[i]
     
def build_cell_translation_vectors(lattice):
    '''
    To overcome the incorrect sample of cell-vector from positions averaging,
    e.g., PBC boundary leads to basis atom move cross boundary (position 0A -> 170A)
    Here directly generate cell-vector for sed based on unitcell-expansion principle
    
    Return:
        cell_vector: 2D np.ndarrys with num_unitcell * 3 elements
    '''
    
    primitive_cell = np.asarray(lattice.dir_lat, dtype=float)
    nx, ny, nz = map(int, np.diagonal(lattice.supercell_matrix))
    translations = []

    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                cell_index = np.array([k, j, i], dtype=float)
                # Each row of primitive_cell is a lattice vector.
                R = cell_index @ primitive_cell
                translations.append(R)

    return np.asarray(translations, dtype=float)


class structure_maker:
    def __init__(self,description='Crystal Lattice'):
        self.description = str(description)

    def basis(self,positions,basis_types,masses=False,reduced_coords=True):
        try:
            self.basis_positions = np.array(positions).astype(float)
        except:
            error('Basis positions must be an N element list of 3D coordinates')

        self.num_basis = self.basis_positions.shape[0]
        if self.basis_positions.shape[1] != 3:
            error('Must provide 3D coordinates for all basis atoms')
        self.reduced_coords = bool(reduced_coords)

        self.basis_types = np.array(basis_types).astype(str)
        if self.basis_types.shape[0] != self.basis_positions.shape[0]:
            error('Number of atom types must match the number of positions given')
        self.unique_types = []
        for atom in basis_types:
            if atom in self.unique_types:
                continue
            else:
                self.unique_types.append(atom)
        print(self.unique_types,flush=True)
        self.num_types = len(self.unique_types)

        if masses != False:
            try:
                self.masses = np.array(masses).astype(float)
            except:
                error('Masses must be floats')
            if len(np.argwhere(self.masses < 0.0001)) != 0:
                error('Masses must be positive')
            if len(self.masses) != self.num_types:
                error('Must provide 1 mass per atom type')
        else:
            self.masses = np.ones(self.num_types)

        self.basis_index = np.arange(1,self.num_basis+1).reshape(self.num_basis,1)
        self.unitcell_index = np.ones((self.num_basis,1))
        self.mass_list = np.zeros((self.num_basis,1))
        for i in range(self.num_types):
            self.mass_list[np.argwhere(self.basis_types == 
                self.unique_types[i])] = self.masses[i]

    def lattice_vectors(self,lattice_vectors,lattice_constants=[1,1,1]):
        try:
            self.lattice_vectors = np.array(lattice_vectors).astype(float)
        except:
            error('Lattice vectors must be a 3 element list of 3D vectors')
        if self.lattice_vectors.shape[0] != 3:
            error('Expected 3 lattice vectors')
        if self.lattice_vectors.shape[1] != 3:
            error('Lattice vectors must be 3D')
        try:
            self.lattice_constants = np.array(lattice_constants).astype(float)
        except:
            error('Lattice constants must be a 3 element list of floats')

        self.lattice_vectors[0,:] = self.lattice_vectors[0,:]*self.lattice_constants[0]
        self.lattice_vectors[1,:] = self.lattice_vectors[1,:]*self.lattice_constants[1]
        self.lattice_vectors[2,:] = self.lattice_vectors[2,:]*self.lattice_constants[2]
        if self.reduced_coords == True:
            for i in range(self.basis_positions.shape[0]):
                self.basis_positions[i,:] = (
                        self.basis_positions[i,0]*self.lattice_vectors[0,:]+
                        self.basis_positions[i,1]*self.lattice_vectors[1,:]+
                        self.basis_positions[i,2]*self.lattice_vectors[2,:])

    def replicate(self,num_reps=[1,1,1]):
        self.num_reps = num_reps
        atom_types = np.copy(self.basis_types)
        pos = np.copy(self.basis_positions[:,:])
        basis = np.copy(self.basis_index)
        mass = np.copy(self.mass_list)
        unitcells = np.copy(self.unitcell_index)

        for i in range(num_reps[0]):
            a1 = np.copy(self.lattice_vectors[0,:])*i
            for j in range(num_reps[1]):
                a2 = np.copy(self.lattice_vectors[1,:])*j
                for k in range(num_reps[2]):
                    a3 = np.copy(self.lattice_vectors[2,:])*k
                    if i == 0 and j == 0 and k == 0: 
                        continue
                    new_cell = np.copy(self.basis_positions)
                    new_cell[:,0] = new_cell[:,0]+a1[0]+a2[0]+a3[0]
                    new_cell[:,1] = new_cell[:,1]+a1[1]+a2[1]+a3[1]
                    new_cell[:,2] = new_cell[:,2]+a1[2]+a2[2]+a3[2]

                    unitcells = unitcells+1
                    self.unitcell_index = np.append(self.unitcell_index,unitcells,axis=0)
                    self.basis_index = np.append(self.basis_index,basis,axis=0)
                    self.mass_list = np.append(self.mass_list,mass,axis=0)

                    pos = np.append(pos,new_cell,axis=0)
                    atom_types = np.append(atom_types,self.basis_types,axis=0)

        pos[:,0] = pos[:,0]-pos[:,0].min()
        pos[:,1] = pos[:,1]-pos[:,1].min()
        pos[:,2] = pos[:,2]-pos[:,2].min()
        self.atom_positions = pos
        self.atom_types = atom_types
        self.num_atoms = pos.shape[0]

    def write_abipos(self,filename='structure.abi'):
        self.unique_types = list(np.unique(self.atom_types))
        self.num_types = len(self.unique_types)
        with open(filename,'w') as fid:
            fid.write(' # ')
            for i in range(self.num_types):
                fid.write('{:d} = {}, '.format(i+1,self.unique_types[i]))
            fid.write('\n\n typat   ')
            for i in range(self.num_atoms):
                fid.write('{:d} '.format(self.unique_types.index(self.atom_types[i])+1))
            fid.write('\n\n xangst   {:10f} {:10f} {:10f}\n'
                    .format(self.atom_positions[0,0],self.atom_positions[0,1],
                        self.atom_positions[0,2]))
            for i in range(1,self.num_atoms):        
                fid.write('          {:10f} {:10f} {:10f}\n'
                        .format(self.atom_positions[i,0],self.atom_positions[i,1],
                        self.atom_positions[i,2]))

    def write_xyz(self,filename='structure.xyz'):
        with open(filename,'w') as fid:
            fid.write('{:d}\n{}\n'.format(self.num_atoms,self.description))
            for i in range(self.num_atoms-1):
                fid.write('{} {:.10f} {:.10f} {:.10f}\n'
                        .format(self.atom_types[i],
                                self.atom_positions[i,0],
                                self.atom_positions[i,1],
                                self.atom_positions[i,2]))
            fid.write('{} {:.10f} {:.10f} {:.10f}\n'
                        .format(self.atom_types[-1],
                                self.atom_positions[-1,0],
                                self.atom_positions[-1,1],
                                self.atom_positions[-1,2]))

    def write_lammps(self,filename='structure.lammps'):
        if (self.lattice_vectors[0,1] != 0 or self.lattice_vectors[0,2] != 0 or
                self.lattice_vectors[1,0] != 0 or self.lattice_vectors[1,2] != 0 or
                self.lattice_vectors[2,0] != 0 or self.lattice_vectors[2,1] != 0):
            error('Can only create LAMMPS file for orthogonal lattice vectors')
            exit()

        with open(filename,'w') as fid:
            fid.write(self.description+'\n\n')
            fid.write('{} atoms\n\n'.format(self.num_atoms))
            fid.write('{} atom types\n\n'.format(self.num_types))
            fid.write('{:12.8f} {:12.8f} xlo xhi\n'.format(0,
                self.lattice_vectors[0,0]*self.num_reps[0]))
            fid.write('{:12.8f} {:12.8f} ylo yhi\n'.format(0,
                self.lattice_vectors[1,1]*self.num_reps[1]))
            fid.write('{:12.8f} {:12.8f} zlo zhi\n\nMasses\n\n'
                    .format(0,self.lattice_vectors[2,2]*self.num_reps[2]))
            for i in range(self.num_types):
                fid.write('{} {:.4f}\n'.format(i+1,self.masses[i]))
            fid.write('\nAtoms\n\n')
            for i in range(self.num_atoms-1):
                fid.write('{} {} {:12.8f} {:12.8f} {:12.8f}\n'
                        .format(i+1,self.unique_types.index(self.atom_types[i])+1,
                        self.atom_positions[i,0],self.atom_positions[i,1],
                        self.atom_positions[i,2]))
            fid.write('{} {} {:12.8f} {:12.8f} {:12.8f}'
                        .format(i+2,self.unique_types.index(self.atom_types[-1])+1,
                        self.atom_positions[-1,0],self.atom_positions[-1,1],
                        self.atom_positions[-1,2]))

    def write_lattice_file(self,filename='lattice.dat'):
        data = np.append(np.arange(1,self.num_atoms+1).reshape(self.num_atoms,1),
                self.unitcell_index,axis=1)
        data = np.append(data,self.basis_index,axis=1)
        data = np.append(data,self.mass_list,axis=1)
        np.savetxt(filename,data,fmt='%d %d %d %.4f')






