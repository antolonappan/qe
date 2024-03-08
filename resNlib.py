
import numpy as np
from plancklens.utils_spin import wignerc


def joincls(cls_list):
    lmaxp1 = np.min([len(cl) for cl in cls_list])
    return np.prod(np.array([cl[:lmaxp1] for cl in cls_list]), axis=0)

def _clinv(cl):
    ret = np.zeros_like(cl)
    ii = np.where(cl != 0)
    ret[ii] = 1./cl[ii]
    return ret

def cli(cl):

    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret


def _dict_transpose(cls):
    ret = {}
    for k in cls.keys():
        if len(k) == 1:
            ret[k + k] = np.copy(cls[k])
        else:
            assert len(k) == 2
            ret[k[1] + k[0]] = np.copy(cls[k])
    return ret

class qeleg:
    def __init__(self, spin_in, spin_out, cl):
        self.spin_in = spin_in
        self.spin_ou = spin_out
        self.cl = cl

    def __eq__(self, leg):
        if self.spin_in != leg.spin_in or self.spin_ou != leg.spin_ou or self.get_lmax() != self.get_lmax():
            return False
        return np.all(self.cl == leg.cl)

    def __mul__(self, other):
        return qeleg(self.spin_in, self.spin_ou, self.cl * other)

    def __add__(self, other):
        assert self.spin_in == other.spin_in and self.spin_ou == other.spin_ou
        lmax = max(self.get_lmax(), other.get_lmax())
        cl = np.zeros(lmax + 1, dtype=float)
        cl[:len(self.cl)] += self.cl
        cl[:len(other.cl)] += other.cl
        return qeleg(self.spin_in, self.spin_ou, cl)

    def copy(self):
        return qeleg(self.spin_in, self.spin_ou, np.copy(self.cl))

    def get_lmax(self):
        return len(self.cl) - 1
    
class qe:
    def __init__(self, leg_a:qeleg, leg_b:qeleg, cL):
        assert leg_a.spin_ou +  leg_b.spin_ou >= 0
        self.leg_a = leg_a
        self.leg_b = leg_b
        self.cL = cL

    def get_lmax_a(self):
        return self.leg_a.get_lmax()

    def get_lmax_b(self):
        return self.leg_b.get_lmax()
    

def qe_simplify(qe_list, _swap=False, verbose=False):
    """Simplifies a list of QE estimators by co-adding terms when possible.


    """
    skip = []
    qes_ret = []
    qes = [qe(q.leg_b.copy(), q.leg_a.copy(), q.cL) for q in qe_list] if _swap else qe_list
    for i, qe1 in enumerate(qes):
        if i not in skip:
            leg_a = qe1.leg_a.copy()
            leg_b = qe1.leg_b.copy()
            for j, qe2 in enumerate(qes[i + 1:]):
                if qe2.leg_a == leg_a:
                    if qe2.leg_b.spin_in == qe1.leg_b.spin_in and qe2.leg_b.spin_ou == qe1.leg_b.spin_ou:
                        Ls = np.arange(max(qe1.leg_b.get_lmax(), qe2.leg_b.get_lmax()) + 1)
                        if np.all(qe1.cL(Ls) == qe2.cL(Ls)):
                            leg_b += qe2.leg_b
                            skip.append(j + i + 1)
            if np.any(leg_a.cl) and np.any(leg_b.cl):
                qes_ret.append(qe(leg_a, leg_b, qe1.cL))
    if verbose and len(skip) > 0:
        print("%s terms down from %s" % (len(qes_ret), len(qes)))
    if not _swap:
        return qe_simplify(qes_ret, _swap=True, verbose=verbose)
    return [qe(q.leg_b.copy(), q.leg_a.copy(), q.cL) for q in qes_ret]

def qe_proj(qe_list, a, b):
    """Projection of a list of QEs onto another QE using only a subset of maps.

        Args:
            qe_list: list of qe instances
            a: (in 't', 'e', 'b') The 1st leg of the output qes will only use this field
            b: (in 't', 'e', 'b') The 2nd leg of the output qes will only use this field
    """
    assert a in ['t', 'e', 'b'] and b in ['t', 'e', 'b']
    l_in = [0] if a == 't' else [-2, 2]
    r_in = [0] if b == 't' else [-2, 2]
    qes_ret = []
    for q in qe_list:
        si, ri = (q.leg_a.spin_in, q.leg_b.spin_in)
        if si in l_in and ri in r_in:
            leg_a = q.leg_a.copy()
            leg_b = q.leg_b.copy()
            if si == 0 and ri == 0:
                qes_ret.append(qe(leg_a, leg_b, q.cL))
            elif si == 0 and abs(ri) > 0:
                sgn = 1 if b == 'e' else -1
                qes_ret.append(qe(leg_a, leg_b * 0.5, q.cL))
                leg_b.spin_in *= -1
                qes_ret.append(qe(leg_a, leg_b * 0.5 * sgn, q.cL))
            elif ri == 0 and abs(si) > 0:
                sgn = 1 if a == 'e' else -1
                qes_ret.append(qe(leg_a * 0.5, leg_b, q.cL))
                leg_a.spin_in *= -1
                qes_ret.append(qe(leg_a * 0.5 * sgn, leg_b, q.cL))
            elif abs(ri) > 0 and abs(si) > 0:
                sgna = 1 if a == 'e' else -1
                sgnb = 1 if b == 'e' else -1
                qes_ret.append(qe(leg_a * 0.5, leg_b * 0.5, q.cL))
                leg_b.spin_in *= -1
                qes_ret.append(qe(leg_a * 0.5, leg_b * 0.5 * sgnb, q.cL))
                leg_a.spin_in *= -1
                qes_ret.append(qe(leg_a * 0.5 * sgna, leg_b * 0.5 * sgnb, q.cL))
                leg_b.spin_in *= -1
                qes_ret.append(qe(leg_a * 0.5 * sgna, leg_b * 0.5, q.cL))
            else:
                assert 0, (si, ri)
    return qe_simplify(qes_ret)

def get_spin_raise(s, lmax):
    r"""Response coefficient of spin-s spherical harmonic to spin raising operator.

        :math:`\sqrt{ (l - s) (l + s + 1) }` for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = np.sqrt(np.arange(abs(s) -s, lmax - s + 1) * np.arange(abs(s) + s + 1, lmax + s + 2))
    return ret

def get_spin_lower(s, lmax):
    r"""Response coefficient of spin-s spherical harmonic to spin lowering operator.

        :math:`-\sqrt{ (l + s) (l - s + 1) }` for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = -np.sqrt(np.arange(s + abs(s), lmax + s + 1) * np.arange(abs(s) - s + 1, lmax - s + 2))
    return ret

def get_resp_legs(source, lmax):
    r"""Defines the responses terms for a CMB map anisotropy source.

    Args:
        source (str): anisotropy source (e.g. 'p', 'f', ...).
        lmax (int): responses are given up to lmax.

    Returns:
        4-tuple (r, rR, -rR, cL):  source spin response *r* (positive or zero),
        the harmonic responses for +r and -r (2 1d-arrays), and the scaling between the G/C modes
        and the potentials of interest. (for lensing, cL is given by :math:`L\sqrt{L (L + 1)}`).

    """
    if source in ['p', 'x']:
        # lensing (gradient and curl): _sX -> _sX -  1/2 alpha_1 \eth _sX - 1/2 \alpha_{-1} \bar \eth _sX
        return {s : (1, -0.5 * get_spin_lower(s, lmax), -0.5 * get_spin_raise(s, lmax),
                     lambda ell : get_spin_raise(0, np.max(ell))[ell]) for s in [0, -2, 2]}

    assert 0, source + ' response legs not implemented'

def spin_cls(s1, s2, cls):
    r"""Spin-weighted power spectrum :math:`_{s1}X_{lm} _{s2}X^{*}_{lm}`

        The output is real unless necessary.


    """
    if s1 < 0:
        return (-1) ** (s1 + s2) * np.conjugate(spin_cls(-s1, -s2, _dict_transpose(cls)))
    assert s1 in [0, -2, 2] and s2 in [0, -2, 2], (s1, s2, 'not implemented')
    if s1 == 0:
        if s2 == 0:
            return cls['tt']
        tb = cls.get('tb', None)
        assert 'te' in cls.keys() or 'et' in cls.keys()
        te = cls.get('te', cls.get('et'))
        return  -te if tb is None else  -te + 1j * np.sign(s2) * tb
    elif s1 == 2:
        if s2 == 0:
            assert 'te' in cls.keys() or 'et' in cls.keys()
            tb = cls.get('bt', cls.get('tb', None))
            et = cls.get('et', cls.get('te'))
            return  -et if tb is None else  -et - 1j * tb
        elif s2 == 2:
            return cls['ee'] + cls['bb']
        elif s2 == -2:
            eb = cls.get('be', cls.get('eb', None))
            return  cls['ee'] - cls['bb'] if eb is None else  cls['ee'] - cls['bb'] + 2j * eb
        else:
            assert 0

def get_covresp(source, s1, s2, cls, lmax, transf=None):
    r"""Defines the responses terms for a CMB covariance anisotropy source.

        \delta < s_d(n) _td^*(n')> \equiv
        _r\alpha(n) W^{r, st}_l _{s - r}Y_{lm}(n) _tY^*_{lm}(n') +
        _r\alpha^*(n') W^{r, ts}_l _{s}Y_{lm}(n) _{t-r}Y^*_{lm}(n')

    """
    if source in ['p','x']:
        # Lensing, modulation, or pol. rotation field from the field representation
        s_source, prR, mrR, cL_scal = get_resp_legs(source, lmax)[s1]
        coupl = spin_cls(s1, s2, cls)[:lmax + 1]
        return s_source, prR * coupl, mrR * coupl, cL_scal
    else:
        assert 0, 'source ' + source + ' cov. response not implemented'

def get_qes(qe_key, lmax, cls_weight, lmax2=None, transf=None):
    """ Defines the quadratic estimator weights for quadratic estimator key.

    Args:
        qe_key (str): quadratic estimator key (e.g., ptt, p_p, ... )
        lmax (int): weights are built up to lmax.
        cls_weight (dict): CMB spectra entering the weights (when relevant).
        lmax2 (int, optional): weight on the second leg are built up to lmax2 (default to lmax)

    The weights are defined by their action on the inverse-variance filtered spin-weight $ _{s}\bar X_{lm}$.

    """
    if lmax2 is None: lmax2 = lmax
    if qe_key[0] in ['p', 'x']:
        if qe_key in ['ptt', 'xtt']:
            s_lefts= [0]
        elif qe_key in ['p_p', 'x_p']:
            s_lefts= [-2, 2]
        else:
            s_lefts = [0, -2, 2]
        qes = []
        s_rights_in = s_lefts
        for s_left in s_lefts:
            for sin in s_rights_in:
                sout = -s_left
                s_qe, irr1, cl_sosi, cL_out =  get_covresp(qe_key[0], sout, sin, cls_weight, lmax2, transf=transf)
                if np.any(cl_sosi):
                    lega = qeleg(s_left, s_left, 0.5 *(1. + (s_left == 0)) * np.ones(lmax + 1, dtype=float))
                    legb = qeleg(sin, sout + s_qe, 0.5 * (1. + (sin == 0)) * 2 * cl_sosi)
                    qes.append(qe(lega, legb, cL_out))
        if len(qe_key) == 1 or qe_key[1:] in ['tt', '_p']:
            return qe_simplify(qes)
        elif qe_key[1:] in ['te', 'et', 'tb', 'bt', 'ee', 'eb', 'be', 'bb']:
            return qe_simplify(qe_proj(qes, qe_key[1], qe_key[2]))
        elif qe_key[1:] in ['_te', '_tb', '_eb']:
            return qe_simplify(qe_proj(qes, qe_key[2], qe_key[3]) + qe_proj(qes, qe_key[3], qe_key[2]))
        else:
            assert 0, 'qe key %s  not recognized'%qe_key
    elif qe_key in ['ntt']:
        lega = qeleg(0, 0, 1   * _clinv(transf[:lmax + 1]))
        legb = qeleg(0, 0, 0.5 * _clinv(transf[:lmax + 1]))  # Weird norm to match PS case for no beam
        qes = [qe(lega, legb, lambda L: np.ones(len(L), dtype=float))]
        return qe_simplify(qes)
    elif qe_key in ['ktt']:
        ls = np.arange(1, lmax + 3)
        dlnDldlnl = ls[:-1] * np.diff(np.log(cls_weight['tt'][ls] * ls * (ls + 1)))
        lega = qeleg(0, 0, np.ones(lmax + 1, dtype=float))
        legb = qeleg(0, 0, 0.5 * cls_weight['tt'][:lmax+1] * dlnDldlnl)
        qes = [qe(lega, legb, lambda L: -L * (L + 1.))]
        return qe_simplify(qes)
    else:
        assert 0, qe_key + ' not implemented'

def get_nhl(qe_key1, qe_key2, cls_weights, cls_ivfs, lmax_ivf1, lmax_ivf2,
            lmax_out=None, lmax_ivf12=None, lmax_ivf22=None, cls_weights2=None,
            cls_ivfs_bb=None, cls_ivfs_ab=None, cls_ivfs_ba=None):
    """(Semi-)Analytical noise level calculation for the cross-spectrum of two QE keys.

        Args:
            qe_key1: QE key 1
            qe_key2: QE key 2
            cls_weights: dictionary with the CMB spectra entering the QE weights.
                        (expected are 'tt', 'te', 'ee' when/if relevant)
            cls_ivfs: dictionary with the inverse-variance filtered CMB spectra.
                        (expected are 'tt', 'te', 'ee', 'bb', 'tb', 'eb' when/if relevant)
            lmax_ivf1: QE 1 uses CMB multipoles down to lmax_ivf1.
            lmax_ivf2: QE 2 uses CMB multipoles down to lmax_ivf2.
            lmax_out(optional): outputs are calculated down to lmax_out. Defaults to lmax_ivf1 + lmax_ivf2
            cls_weights2(optional): Second QE cls weights, if different from cls_weights

        Outputs:
            4-tuple of gradient (G) and curl (C) mode Gaussian noise co-variances GG, CC, GC, CG.

    """
    if lmax_ivf12 is None: lmax_ivf12 = lmax_ivf1
    if lmax_ivf22 is None: lmax_ivf22 = lmax_ivf2
    if cls_weights2 is None: cls_weights2 = cls_weights
    qes1 = get_qes(qe_key1, lmax_ivf1, cls_weights, lmax2=lmax_ivf12)
    qes2 = get_qes(qe_key2, lmax_ivf2, cls_weights2, lmax2=lmax_ivf22)
    if lmax_out is None:
        lmax_out = max(lmax_ivf1, lmax_ivf12) + max(lmax_ivf2, lmax_ivf22)
    return  _get_nhl(qes1, qes2, cls_ivfs, lmax_out, cls_ivfs_bb=cls_ivfs_bb, cls_ivfs_ab=cls_ivfs_ab, cls_ivfs_ba=cls_ivfs_ba)


def _get_nhl(qes1, qes2, cls_ivfs, lmax_out, cls_ivfs_bb=None, cls_ivfs_ab=None, cls_ivfs_ba=None, ret_terms=False):
    GG_N0 = np.zeros(lmax_out + 1, dtype=float)
    CC_N0 = np.zeros(lmax_out + 1, dtype=float)
    GC_N0 = np.zeros(lmax_out + 1, dtype=float)
    CG_N0 = np.zeros(lmax_out + 1, dtype=float)

    cls_ivfs_aa = cls_ivfs
    cls_ivfs_bb = cls_ivfs if cls_ivfs_bb is None else cls_ivfs_bb
    cls_ivfs_ab = cls_ivfs if cls_ivfs_ab is None else cls_ivfs_ab
    cls_ivfs_ba = cls_ivfs if cls_ivfs_ba is None else cls_ivfs_ba
    if ret_terms:
        terms = []
    for qe1 in qes1:
        cL1 = qe1.cL(np.arange(lmax_out + 1))
        for qe2 in qes2:
            cL2 = qe2.cL(np.arange(lmax_out + 1))
            si, ti, ui, vi = (qe1.leg_a.spin_in, qe1.leg_b.spin_in, qe2.leg_a.spin_in, qe2.leg_b.spin_in)
            so, to, uo, vo = (qe1.leg_a.spin_ou, qe1.leg_b.spin_ou, qe2.leg_a.spin_ou, qe2.leg_b.spin_ou)
            assert so + to >= 0 and uo + vo >= 0, (so, to, uo, vo)

            clsu = joincls([qe1.leg_a.cl, qe2.leg_a.cl.conj(), spin_cls(si, ui, cls_ivfs_aa)])
            cltv = joincls([qe1.leg_b.cl, qe2.leg_b.cl.conj(), spin_cls(ti, vi, cls_ivfs_bb)])
            R_sutv = joincls([wignerc(clsu, cltv, so, uo, to, vo, lmax_out=lmax_out), cL1, cL2])

            clsv = joincls([qe1.leg_a.cl, qe2.leg_b.cl.conj(), spin_cls(si, vi, cls_ivfs_ab)])
            cltu = joincls([qe1.leg_b.cl, qe2.leg_a.cl.conj(), spin_cls(ti, ui, cls_ivfs_ba)])
            R_sutv = R_sutv + joincls([wignerc(clsv, cltu, so, vo, to, uo, lmax_out=lmax_out), cL1, cL2])

            # we now need -s-t uv
            sgnms = (-1) ** (si + so)
            sgnmt = (-1) ** (ti + to)
            clsu = joincls([sgnms * qe1.leg_a.cl.conj(), qe2.leg_a.cl.conj(), spin_cls(-si, ui, cls_ivfs_aa)])
            cltv = joincls([sgnmt * qe1.leg_b.cl.conj(), qe2.leg_b.cl.conj(), spin_cls(-ti, vi, cls_ivfs_bb)])
            R_msmtuv = joincls([wignerc(clsu, cltv, -so, uo, -to, vo, lmax_out=lmax_out), cL1, cL2])

            clsv = joincls([sgnms * qe1.leg_a.cl.conj(), qe2.leg_b.cl.conj(), spin_cls(-si, vi, cls_ivfs_ab)])
            cltu = joincls([sgnmt * qe1.leg_b.cl.conj(), qe2.leg_a.cl.conj(), spin_cls(-ti, ui, cls_ivfs_ba)])
            R_msmtuv = R_msmtuv + joincls([wignerc(clsv, cltu, -so, vo, -to, uo, lmax_out=lmax_out), cL1, cL2])

            GG_N0 +=  0.5 * R_sutv.real
            GG_N0 +=  0.5 * (-1) ** (to + so) * R_msmtuv.real

            CC_N0 += 0.5 * R_sutv.real
            CC_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.real

            GC_N0 -= 0.5 * R_sutv.imag
            GC_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.imag

            CG_N0 += 0.5 * R_sutv.imag
            CG_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.imag
            if ret_terms:
                terms += [0.5 * R_sutv, 0.5 * (-1) ** (to + so) * R_msmtuv]
    return (GG_N0, CC_N0, GC_N0, CG_N0) if not ret_terms else (GG_N0, CC_N0, GC_N0, CG_N0, terms)


def get_response(qe_key, lmax_ivf, source, cls_weight, cls_cmb, fal, fal_leg2=None, lmax_ivf2=None, lmax_qlm=None, transf=None):
    r"""QE response calculation

        Args:
            qe_key: Quadratic estimator key (see this module docstring for descriptions)
            lmax_ivf: max. CMB multipole used in the QE
            source: anisotropy source key
            cls_weight(dict): fiducial spectra entering the QE weights (numerator in Eq. 2 of https://arxiv.org/abs/1807.06210)
            cls_cmb(dict): CMB spectra entering the CMB response (in principle lensed spectra, or grad-lensed spectra)
            fal(dict): (isotropic approximation to the) filtering cls. e.g. fal['tt'] :math:`= \frac {1} {C^{\rm TT}_\ell  +  N^{\rm TT}_\ell / b^2_\ell}` for temperature if filtered independently from polarization.
            fal_leg2(dict): Same as *fal* but for the second leg, if different.
            lmax_ivf2(optional): max. CMB multipole used in the QE on the second leg (if different to lmax_ivf)
            lmax_qlm(optional): responses are calculated up to this multipole. Defaults to lmax_ivf + lmax_ivf2

        Note:
            The result is *not* symmetrized with respect to the 'fals', if not the same on the two legs.
            In this case you probably want to run this twice swapping the fals in the second run.

    """
    if lmax_ivf2 is None: lmax_ivf2 = lmax_ivf
    if lmax_qlm is None : lmax_qlm = lmax_ivf + lmax_ivf2


    qes = get_qes(qe_key, lmax_ivf, cls_weight, lmax2=lmax_ivf2, transf=transf)

    return _get_response(qes, source, cls_cmb, fal, lmax_qlm, fal_leg2=fal_leg2)

    
def _get_response(qes, source, cls_cmb, fal_leg1, lmax_qlm, fal_leg2=None):
    fal_leg2 = fal_leg1 if fal_leg2 is None else fal_leg2
    RGG = np.zeros(lmax_qlm + 1, dtype=float)
    RCC = np.zeros(lmax_qlm + 1, dtype=float)
    RGC = np.zeros(lmax_qlm + 1, dtype=float)
    RCG = np.zeros(lmax_qlm + 1, dtype=float)
    Ls = np.arange(lmax_qlm + 1, dtype=int)
    for qe in qes:
        si, ti = (qe.leg_a.spin_in, qe.leg_b.spin_in)
        so, to = (qe.leg_a.spin_ou, qe.leg_b.spin_ou)
        for s2 in ([0, -2, 2]):
            FA = get_spin_matrix(si, s2, fal_leg1)
            if np.any(FA):
                for t2 in ([0, -2, 2]):
                    FB = get_spin_matrix(ti, t2, fal_leg2)
                    if np.any(FB):
                        rW_st, prW_st, mrW_st, s_cL_st = get_covresp(source, -s2, t2, cls_cmb, len(FB) - 1)
                        clA = joincls([qe.leg_a.cl, FA])
                        clB = joincls([qe.leg_b.cl, FB, mrW_st.conj()])
                        Rpr_st = wignerc(clA, clB, so, s2, to, -s2 + rW_st, lmax_out=lmax_qlm) * s_cL_st(Ls)

                        rW_ts, prW_ts, mrW_ts, s_cL_ts = get_covresp(source, -t2, s2, cls_cmb, len(FA) - 1)
                        clA = joincls([qe.leg_a.cl, FA, mrW_ts.conj()])
                        clB = joincls([qe.leg_b.cl, FB])
                        Rpr_st = Rpr_st + wignerc(clA, clB, so, -t2 + rW_ts, to, t2, lmax_out=lmax_qlm) * s_cL_ts(Ls)
                        assert rW_st == rW_ts and rW_st >= 0, (rW_st, rW_ts)
                        if rW_st > 0:
                            clA = joincls([qe.leg_a.cl, FA])
                            clB = joincls([qe.leg_b.cl, FB, prW_st.conj()])
                            Rmr_st = wignerc(clA, clB, so, s2, to, -s2 - rW_st, lmax_out=lmax_qlm) * s_cL_st(Ls)

                            clA = joincls([qe.leg_a.cl, FA, prW_ts.conj()])
                            clB = joincls([qe.leg_b.cl, FB])
                            Rmr_st = Rmr_st + wignerc(clA, clB, so, -t2 - rW_ts, to, t2, lmax_out=lmax_qlm) * s_cL_ts(Ls)
                        else:
                            Rmr_st = Rpr_st
                        prefac = qe.cL(Ls)
                        RGG += prefac * ( Rpr_st.real + Rmr_st.real * (-1) ** rW_st)
                        RCC += prefac * ( Rpr_st.real - Rmr_st.real * (-1) ** rW_st)
                        RGC += prefac * (-Rpr_st.imag + Rmr_st.imag * (-1) ** rW_st)
                        RCG += prefac * ( Rpr_st.imag + Rmr_st.imag * (-1) ** rW_st)

    return RGG, RCC, RGC, RCG

def get_spin_matrix(sout, sin, cls):
    r"""Spin-space matrix R^{-1} cls[T, E, B] R where R is the mapping from _{0, \pm 2}X to T, E, B.

        cls is dictionary with keys 'tt', 'te', 'ee', 'bb'.
        If a key is not present the corresponding spectrum is assumed to be zero.
        ('t' 'e' and 'b' keys also works in place of 'tt' 'ee', 'bb'.)

        Output is complex only when necessary (that is, TB and/or EB present and relevant).

    """
    assert sin in [0, 2, -2] and sout in [0, 2, -2], (sin, sout)
    if sin == 0:
        if sout == 0:
            return cls.get('tt', cls.get('t', 0.))
        tb = cls.get('tb', None)
        return (-cls.get('te', 0.) - 1j * np.sign(sout) * tb) if tb is not None else -cls.get('te', 0.)
    if sin == 2:
        if sout == 0:
            te = cls.get('te', 0.)
            tb = cls.get('tb', None)
            return -0.5 * (te - 1j * tb) if tb is not None else -0.5 * te
        if sout == 2:
            return 0.5 * (cls.get('ee', cls.get('e', 0.)) + cls.get('bb', cls.get('b', 0.)))
        if sout == -2:
            ret =  0.5 * (cls.get('ee', cls.get('e', 0.)) - cls.get('bb', cls.get('b', 0.)))
            eb = cls.get('eb', None)
            return ret - 1j * eb if eb is not None else ret
    if sin == -2:
        if sout == 0:
            te = cls.get('te', 0.)
            tb = cls.get('tb', None)
            return -0.5 * (te + 1j * tb) if tb is not None else -0.5 * te
        if sout == 2:
            ret =  0.5 * (cls.get('ee', cls.get('e', 0.)) - cls.get('bb', cls.get('b', 0.)))
            eb = cls.get('eb', None)
            return ret + 1j * eb if eb is not None else ret
        if sout == -2:
            return 0.5 * (cls.get('ee', cls.get('e', 0.)) + cls.get('bb', cls.get('b', 0.)))
    assert 0, (sin, sout)