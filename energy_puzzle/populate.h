#pragma once
#include "fields.h"

void populate(){
  const Field I =
    make_energy_mat(
      "MDDDEDDDMDDDEDDDMDDDCDDDMDDDEDDDMDDDEDDDM",
      OpList{
      {'E','C',1.},
      {'E','E',12.},
      {'M','M',-4.},
    });
  letters['I'] = I;
  const Field A =
    make_energy_mat(
      "EDDDLDDDTDDDRDDDEDDDRDDL",
      OpList{
        {'T','E',100},
        {'L','R',1},
        {'L','L',-4},
        {'R','R',-4},
    });
  letters['A'] = A;
  const Field T =
    make_energy_mat(
      "EDDMDDDDDBDDDDDMDDE",
      {
        {'M','M',-50},
        {'E','E',10},
        {'E','B',1},
    });
  letters['T'] = T;
  const Field B =
    make_energy_mat(
      "TDDMDDBDDRDDMDDRDDT",
      {
        {'T','T',-100},
        {'M','M',-100},
        {'T','B',10},
        {'R','R',-0.1},
    });
  letters['B'] = B;
  const Field O =
    make_energy_mat(
      "TDDDDDDDDDDDDDDDT",
      {
        {'T','T',-100},
        {'D','D',1},
    });
  letters['O'] = O;
  const Field L =
    make_energy_mat(
      "TDDDDMBMBMRMDDDDE",
      {
        {'T','B',100},
        {'M','M',-200},
        {'B','E',10},
        {'R','B',10},
        {'R','E',1},
    });
  letters['L'] = L;
}
