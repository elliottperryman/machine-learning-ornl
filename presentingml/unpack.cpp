#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>
using namespace std;


int main(int argc, char* argv[]) {
	string name(argv[1]);
	
	// setup reading in	
	//ifstream fin(name.c_str(), ios::binary);
	//if (!fin.good()) perror("fin");
	FILE* fin = fopen(name.c_str(), "rb");

	// setup reading out
	name.erase(name.length()-3, name.length()-1);
	name.append("csv");	
	ofstream fout(name);
	if (!fout.good()) perror("fout");
//	FILE* fout = fopen(name.c_str(), "w");

	short tmp[3500];
	//fin.seekg(8, fin.beg);	
	fseek(fin, 8, SEEK_SET);	
	for (int j=0;j<1000;j++) {
		//fin.seekg(33,fin.cur);
		fseek(fin, 33, SEEK_CUR);	
		fread(tmp, 2, 3500, fin);	
		for (int i=0;i<3499;i++) {
			fout<<tmp[i]<<',';
		}
		fout<<tmp[3499];
		fout<<"\n";	
	}

	return 1;
}
