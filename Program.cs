//algorytm wstecznej propagacji 2 (XOR + NOR) | Jacek Domeracki | numer albumu: 173518

using System;
using System.Collections.Generic;
using System.Linq;

namespace Zadanie_5
{
    public class Neuron
    {
        public List<double> lista_wartosci_wag = new List<double>();
        public double wartosc_wyjscia = 0;
        public double korekta_wyjscia = 0;
        public double korekta_sumy_wewa = 0;
        public List<double> lista_korekt_wag = new List<double>();
        public List<double> lista_korekt_wejsc = new List<double>();
    }

    public class Siec_neuronowa         //warstwa nr 0 to wejścia sieci, waga nr 0 to bias neuronu
    {
        private List<Neuron>[] warstwy_neurony;

        public Siec_neuronowa(int[] licznosc_warstw)
        {
            warstwy_neurony = new List<Neuron>[licznosc_warstw.Length];
            for (int i = 0; i < licznosc_warstw.Length; i++)
            {
                warstwy_neurony[i] = new List<Neuron>();
                for (int j = 0; j < licznosc_warstw[i]; j++)
                {
                    warstwy_neurony[i].Add(new Neuron());
                    if (i == 0)
                        continue;
                    for (int k = 0; k < licznosc_warstw[i - 1] + 1; k++)        //tyle wag neuronu warstwy i, ile neuronów warstwy i-1, plus bias
                    {
                        warstwy_neurony[i][j].lista_wartosci_wag.Add(0);
                        warstwy_neurony[i][j].lista_korekt_wag.Add(0);
                        if (i == 1)
                            continue;
                        warstwy_neurony[i][j].lista_korekt_wejsc.Add(0);
                    }
                }
            }
        }

        public void Ustaw_losowe_wartosci_wag(double przedz_min, double przedz_max)
        {
            Random random = new Random();
            for (int i = 1; i < warstwy_neurony.Length; i++)
            {
                for (int j = 0; j < warstwy_neurony[i].Count; j++)
                {
                    for (int k = 0; k < warstwy_neurony[i][j].lista_wartosci_wag.Count; k++)
                    {
                        warstwy_neurony[i][j].lista_wartosci_wag[k] = random.NextDouble() * (przedz_max - przedz_min) + przedz_min;
                    }
                }
            }
        }

        private double Funkcja_aktywacji(double beta, double wartosc_bez_fa)
        {
            return 1 / (1 + Math.Exp(-beta * wartosc_bez_fa));
        }

        public void Oblicz_wyjscie_sieci_neuronowej(List<double> wej_probka, double beta, ref List<double> wartosc_wyjscia_sieci)
        {
            for (int i = 0; i < wej_probka.Count; i++)
            {
                warstwy_neurony[0][i].wartosc_wyjscia = wej_probka[i];
            }

            for (int i = 1; i < warstwy_neurony.Length; i++)
            {
                for (int j = 0; j < warstwy_neurony[i].Count; j++)
                {
                    double sum_wart_wyj = warstwy_neurony[i][j].lista_wartosci_wag[0];
                    for (int k = 1; k < warstwy_neurony[i][j].lista_wartosci_wag.Count; k++)
                    {
                        sum_wart_wyj += warstwy_neurony[i - 1][k - 1].wartosc_wyjscia * warstwy_neurony[i][j].lista_wartosci_wag[k];
                    }
                    warstwy_neurony[i][j].wartosc_wyjscia = Funkcja_aktywacji(beta, sum_wart_wyj);
                }
            }
            for (int i = 0; i < wartosc_wyjscia_sieci.Count; i++)
            {
                wartosc_wyjscia_sieci[i] = warstwy_neurony[warstwy_neurony.Length - 1][i].wartosc_wyjscia;

            }
        }

        public void Propaguj_wstecznie_korekty(List<double> wyj_probka, double beta, double mi)
        {
            for (int i = 0; i < wyj_probka.Count; i++)
            {
                warstwy_neurony[warstwy_neurony.Length - 1][i].korekta_wyjscia = mi * (wyj_probka[i] - warstwy_neurony[warstwy_neurony.Length - 1][i].wartosc_wyjscia);
            }

            for (int i = warstwy_neurony.Length - 1; i > 0; i--)
            {
                for (int j = 0; j < warstwy_neurony[i].Count; j++)
                {
                    warstwy_neurony[i][j].korekta_sumy_wewa = warstwy_neurony[i][j].korekta_wyjscia
                                                    * beta * warstwy_neurony[i][j].wartosc_wyjscia * (1 - warstwy_neurony[i][j].wartosc_wyjscia);

                    warstwy_neurony[i][j].lista_korekt_wag[0] = warstwy_neurony[i][j].korekta_sumy_wewa * 1;
                    for (int k = 1; k < warstwy_neurony[i][j].lista_korekt_wag.Count; k++)
                    {
                        warstwy_neurony[i][j].lista_korekt_wag[k] = warstwy_neurony[i][j].korekta_sumy_wewa * warstwy_neurony[i - 1][k - 1].wartosc_wyjscia;
                    }
                    if (i == 1)
                        continue;
                    for (int k = 1; k < warstwy_neurony[i][j].lista_korekt_wejsc.Count; k++)
                    {
                        warstwy_neurony[i][j].lista_korekt_wejsc[k] = warstwy_neurony[i][j].korekta_sumy_wewa * warstwy_neurony[i][j].lista_wartosci_wag[k];
                    }
                }
                if (i == 1)
                    continue;
                for (int j = 0; j < warstwy_neurony[i - 1].Count; j++)
                {
                    warstwy_neurony[i - 1][j].korekta_wyjscia = 0;
                    for (int k = 0; k < warstwy_neurony[i].Count; k++)
                    {
                        warstwy_neurony[i - 1][j].korekta_wyjscia += warstwy_neurony[i][k].lista_korekt_wejsc[j + 1];
                    }
                }
            }
            for (int i = 1; i < warstwy_neurony.Length; i++)            //korekta wszystkich wag sieci
            {
                for (int j = 0; j < warstwy_neurony[i].Count; j++)
                {
                    for (int k = 0; k < warstwy_neurony[i][j].lista_korekt_wag.Count; k++)
                    {
                        warstwy_neurony[i][j].lista_wartosci_wag[k] += warstwy_neurony[i][j].lista_korekt_wag[k];
                    }
                }
            }
        }

        public void Wypisz_wagi_strukturalnie()
        {
            Console.WriteLine("--------------------------------------------------");
            for (int i = 1; i < warstwy_neurony.Length; i++)
            {
                Console.WriteLine("WARSTWA NR : {0}", i);
                for (int j = 0; j < warstwy_neurony[i].Count; j++)
                {
                    Console.WriteLine("    NEURON NR : {0}", j);
                    for (int k = 0; k < warstwy_neurony[i][j].lista_wartosci_wag.Count; k++)
                    {
                        Console.WriteLine("        WAGA NR : {0}  |  WAGA : {1,10:F6}", k, warstwy_neurony[i][j].lista_wartosci_wag[k]);
                    }
                }
            }
            Console.WriteLine("--------------------------------------------------");
        }
    }

    internal class Program
    {
        static readonly int[] SCHEMAT_SIECI_NEURONOWEJ = { 2, 2, 2, 2 };
        const double PRZEDZ_MIN = -5;
        const double PRZEDZ_MAX = 5;
        const int ILE_ITERACJE = 50000;
        const int ILE_WID_GD = 200;         //ile widocznych iteracji z góry i z dołu
        const double BETA = 1;
        const double MI = 0.3;
        const double MARGINES_BLEDU = 0.4;

        static void Main()
        {
            Console.WriteLine("ALGORYTM WSTECZNEJ PROPAGACJI ( ZADANIE 2 )");
            Console.WriteLine();

            List<Tuple<List<double>, List<double>>> Probki_funkcji_xornor = new List<Tuple<List<double>, List<double>>>
            {
                new Tuple<List<double>, List<double>> ( new List<double> { 0, 0 }, new List<double> { 0, 1 } ),
                new Tuple<List<double>, List<double>> ( new List<double> { 0, 1 }, new List<double> { 1, 0 } ),
                new Tuple<List<double>, List<double>> ( new List<double> { 1, 0 }, new List<double> { 1, 0 } ),
                new Tuple<List<double>, List<double>> ( new List<double> { 1, 1 }, new List<double> { 0, 0 } )
            };
            Tuple<List<double>, List<double>> probka_rob;

            Siec_neuronowa SN = new Siec_neuronowa(SCHEMAT_SIECI_NEURONOWEJ);
            SN.Ustaw_losowe_wartosci_wag(PRZEDZ_MIN, PRZEDZ_MAX);

            Console.WriteLine("-->  START");
            Console.WriteLine();

            Random random = new Random();
            List<int> indeksy_probek = new List<int>();
            int n_pr;

            List<double> wartosc_wyjscia_SN = new List<double>(Probki_funkcji_xornor[0].Item2);             //muszą być zainicjowane
            List<double> nowa_wartosc_wyjscia_SN = new List<double>(Probki_funkcji_xornor[0].Item2);
            List<double> blizej_do_probki_wyj = new List<double>(Probki_funkcji_xornor[0].Item2);

            for (int i = 0; i < ILE_ITERACJE; i++)
            {
                if (i < ILE_WID_GD || i >= ILE_ITERACJE - ILE_WID_GD) Console.WriteLine("-->  EPOKA NR : {0}", i + 1);

                for (int j = 0; j < Probki_funkcji_xornor.Count; j++)
                {
                    indeksy_probek.Add(j);
                }
                for (int j = 0; j < Probki_funkcji_xornor.Count; j++)
                {
                    n_pr = indeksy_probek[random.Next(indeksy_probek.Count)];
                    indeksy_probek.Remove(n_pr);

                    probka_rob = Probki_funkcji_xornor[n_pr];
                    SN.Oblicz_wyjscie_sieci_neuronowej(probka_rob.Item1, BETA, ref wartosc_wyjscia_SN);

                    SN.Propaguj_wstecznie_korekty(probka_rob.Item2, BETA, MI);
                    SN.Oblicz_wyjscie_sieci_neuronowej(probka_rob.Item1, BETA, ref nowa_wartosc_wyjscia_SN);

                    for (int k = 0; k < probka_rob.Item2.Count; k++)
                    {
                        blizej_do_probki_wyj[k] = Math.Abs(probka_rob.Item2[k] - wartosc_wyjscia_SN[k]) - Math.Abs(probka_rob.Item2[k] - nowa_wartosc_wyjscia_SN[k]);
                    }

                    if (i < ILE_WID_GD || i >= ILE_ITERACJE - ILE_WID_GD)
                    {
                        for (int k = 0; k < probka_rob.Item2.Count; k++)
                        {
                            Console.WriteLine("------>  PRÓBKA NR : {0}  |  NOWE WYJŚCIE : {1,8:F6}  |  UBYTEK BŁĘDU : {2,12:F10}  |  NOWY BŁĄD: {3,9:F6}",
                                n_pr + 1, nowa_wartosc_wyjscia_SN[k], blizej_do_probki_wyj[k], probka_rob.Item2[k] - nowa_wartosc_wyjscia_SN[k]);
                        }
                    }
                }
                if (i < ILE_WID_GD || i >= ILE_ITERACJE - ILE_WID_GD) Console.WriteLine();
            }

            n_pr = 0;
            foreach (var probka in Probki_funkcji_xornor)
            {
                SN.Oblicz_wyjscie_sieci_neuronowej(probka.Item1, BETA, ref wartosc_wyjscia_SN);

                for (int i = 0; i < probka.Item2.Count; i++)
                {
                    if (Math.Abs(probka.Item2[i] - wartosc_wyjscia_SN[i]) >= MARGINES_BLEDU)
                    {
                        n_pr++;
                    }
                }
            }
            Console.WriteLine("---------->  MARGINES BŁĘDU : {0,6:F4}", MARGINES_BLEDU);
            if (n_pr == 0)
            {
                Console.WriteLine("---------->  SPEŁNIONY");
                SN.Wypisz_wagi_strukturalnie();
            }
            else
            {
                Console.WriteLine("/\\/\\------>  NIESPEŁNIONY : {0} -KROTNIE", n_pr);
            }
            Console.WriteLine();

            Console.WriteLine("-->  KONIEC");
            Console.WriteLine();
        }
    }
}
