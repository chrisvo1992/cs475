#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// print debugging messages?
#ifndef DEBUG
#define DEBUG   false
#endif

// setting the number of threads:
#ifndef NUMT
#define NUMT                2
#endif

// setting the number of trials in the monte carlo simulation:
#ifndef NUMTRIALS
#define NUMTRIALS       50000
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTIMES
#define NUMTIMES        20
#endif

// ranges for the random numbers:
const float TXMIN = -10;
const float TXMAX = 10;
const float TXVMIN = 10;
const float TXVMAX = 30;
const float TYMIN = 45;
const float TYMAX = 55;
const float SVMIN = 10;
const float SVMAX = 30;
const float STHMIN = 10;
const float STHMAX = 90;


// degrees-to-radians:
inline
float Radians( float d )
{
	        return (M_PI/180.f) * d;
}

void
TimeOfDaySeed( )
{
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time( &timer );
	double seconds = difftime( timer, mktime(&y2k) );
	unsigned int seed = (unsigned int)( 1000.*seconds );
	srand( seed );
}
float
Ranf( float low, float high ){
	float r = (float) rand();
	float t = r  /  (float) RAND_MAX;
	return   low  +  t * ( high - low );
}

int
Ranf( int ilow, int ihigh )
{
	float low = (float)ilow;
	float high = ceil( (float)ihigh );
	return (int) Ranf(low,high);
}

int
main( int argc, char *argv[ ] )
{
#ifndef _OPENMP
	fprintf( stderr, "No OpenMP support!\n" );
	return 1;
#endif
        TimeOfDaySeed( );
	omp_set_num_threads( NUMT );
	float *txs  = new float [NUMTRIALS];
	float *tys  = new float [NUMTRIALS];
	float *txvs = new float [NUMTRIALS];
	float *svs  = new float [NUMTRIALS];
	float *sths = new float [NUMTRIALS];

	for( int n = 0; n < NUMTRIALS; n++ ){
		txs[n]  = Ranf(  TXMIN,  TXMAX );
		tys[n]  = Ranf(  TYMIN,  TYMAX );
		txvs[n] = Ranf(  TXVMIN, TXVMAX );
		svs[n]  = Ranf(  SVMIN,  SVMAX );
		sths[n] = Ranf(  STHMIN, STHMAX );
	}
// get ready to record the maximum performance and the probability:
	double  maxPerformance = 0.;
	int     numHits;
	for( int times = 0; times < NUMTIMES; times++ ){
		double time0 = omp_get_wtime( );
		numHits = 0;
		#pragma omp parallel for default(none) shared(txs,tys,txvs,svs,sths,stderr) reduction(+:numHits)
		for( int n = 0; n < NUMTRIALS; n++ ){
			float tx   = txs[n];
			float ty   = tys[n];
			float txv  = txvs[n];
			float sv   = svs[n];
			float sthd = sths[n];
			float sthr = Radians( sthd );
			float svx  = sv * cos(sthr);
			float svy  = sv * sin(sthr);
			// how long until the snowball reaches the y depth:
			float t = ty/svy;
			float truckx = tx+txv*t;
			float sbx = svx*t;

			if( fabs(truckx - sbx) < 20 ){
				numHits++;
				if( DEBUG )  fprintf( stderr, "Hits the truck at time = %8.3f\n", t );
			}
		}
		double time1 = omp_get_wtime( );
		double megaTrialsPerSecond = (double)NUMTRIALS / ( time1 - time0 ) / 1000000.;
		if( megaTrialsPerSecond > maxPerformance )
			maxPerformance = megaTrialsPerSecond;
	}
	float probability = (float)numHits/(float)( NUMTRIALS );
	fprintf(stderr, "%2d threads : %8d trials ; probability = %6.2f%% ; megatrials/sec = %6.2lf\n",
			NUMT, NUMTRIALS, 100.*probability, maxPerformance);
	return 0;
}
