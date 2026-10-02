"""Tests unitaires des fonctions de calcul."""

import pytest

from processing.build_gold import trimp
from processing.build_silver import dig


def test_trimp_nul_au_repos():
    # Une FC moyenne égale à la FC de repos ne génère aucune charge
    assert trimp(60, avg_hr=50, hr_rest=50, hr_max=190) == 0


def test_trimp_augmente_avec_la_duree():
    court = trimp(30, avg_hr=150, hr_rest=50, hr_max=190)
    long = trimp(60, avg_hr=150, hr_rest=50, hr_max=190)
    assert long == pytest.approx(2 * court)


def test_trimp_augmente_avec_l_intensite():
    facile = trimp(60, avg_hr=120, hr_rest=50, hr_max=190)
    dur = trimp(60, avg_hr=170, hr_rest=50, hr_max=190)
    assert dur > facile


def test_trimp_borne_si_fc_hors_limites():
    # Une FC aberrante (sous le repos ou au-dessus du max) ne doit pas exploser
    assert trimp(60, avg_hr=40, hr_rest=50, hr_max=190) == 0
    assert trimp(60, avg_hr=250, hr_rest=50, hr_max=190) == trimp(60, 190, 50, 190)


def test_dig_lit_une_valeur_imbriquee():
    data = {"a": {"b": {"c": 42}}}
    assert dig(data, "a", "b", "c") == 42


def test_dig_renvoie_none_si_cle_absente():
    assert dig({"a": {}}, "a", "b", "c") is None
    assert dig(None, "a") is None
