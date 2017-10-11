/*
 * task.h
 *
 *  Created on: Oct 10, 2017
 *      Author: chaoliu
 */

#ifndef TASK_H_
#define TASK_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "Utc.h"
#include "typeconfig.h"

class SceneInit: public UserTaskBase{
public:
	void runImpl(char* filename,
			sphere2_t **obj_array_in,
			global_vars *g_vars,
			vec3_t *lights ){
		FILE *fp = fopen((const char*)filename, "rb");
		if(fp ==NULL){
			std::cerr<<"Need a input scene file."<<std::endl;
			exit(1);
		}

		/*
		 * read scene file
		 */
		char line[256], *ptr, type;

		sphere_t *obj_list = (sphere_t*)malloc(sizeof(struct sphere));
		obj_list->next = NULL;
		int obj_count=0;
		camera_t cam;
		int lnum=0;
#define DELIM	" \t\n"
		while((ptr = fgets(line, 256, fp))) {
			int i;
			vec3_t pos, col;
			FTYPE rad, spow, refl;

			while(*ptr == ' ' || *ptr == '\t') ptr++;
			if(*ptr == '#' || *ptr == '\n') continue;

			if(!(ptr = strtok(line, DELIM))) continue;
			type = *ptr;

			for(i=0; i<3; i++) {
				if(!(ptr = strtok(0, DELIM))) break;
				*((FTYPE*)&pos.x + i) = (FTYPE)atof(ptr);
			}

			if(type == 'l') {
				lights[lnum++] = pos;
				continue;
			}

			if(!(ptr = strtok(0, DELIM))) continue;
			rad = atof(ptr);

			for(i=0; i<3; i++) {
				if(!(ptr = strtok(0, DELIM))) break;
				*((FTYPE*)&col.x + i) = (FTYPE)atof(ptr);
			}

			if(type == 'c') {
				cam.pos = pos;
				cam.targ = col;
				cam.fov = rad;
				continue;
			}

			if(!(ptr = strtok(0, DELIM))) continue;
			spow = (FTYPE)atof(ptr);

			if(!(ptr = strtok(0, DELIM))) continue;
			refl = (FTYPE)atof(ptr);

			if(type == 's') {
				obj_count++;
				struct sphere *sph = (sphere_t*)malloc(sizeof *sph);
				sph->next = obj_list->next;
				obj_list->next = sph;

				sph->pos = pos;
				sph->rad = rad;
				sph->mat.col = col;
				sph->mat.spow = spow;
				sph->mat.refl = refl;
			} else {
				fprintf(stderr, "unknown type: %c\n", type);
			}
		}
		fclose(fp);
		/*
		 * change the sphere linked list to an array
		 */
		sphere2_t *obj_array = (sphere2_t*)malloc(obj_count * sizeof(sphere2_t));
		sphere_t *p1 = obj_list->next;
		sphere_t *p2 = p1;
		int i=0;
		while(p1!=nullptr){
			obj_array[i].pos = p1->pos;
			obj_array[i].rad = p1->rad;
			obj_array[i].mat.col = p1->mat.col;
			obj_array[i].mat.spow = p1->mat.spow;
			obj_array[i].mat.refl = p1->mat.refl;
			p2 = p1;
			p1 = p1->next;
			free(p2);
			i++;
		}
		obj_list->next = NULL;
		free(obj_list);
		*obj_array_in = obj_array;

		g_vars->cam = cam;
		g_vars->lnum = lnum;
		g_vars->obj_count = obj_count;
	}
};


class Output: public UserTaskBase{
public:
	void runImpl(char* filename, uint32_t *pixels, int xres, int yres){
		FILE *outfile = fopen((const char*)filename, "wb");
		if(outfile != NULL){
			fprintf(outfile, "P6\n%d %d\n255\n", xres, yres);
			for(int i=0; i<xres * yres; i++) {
				fputc((pixels[i] >> RSHIFT) & 0xff, outfile);
				fputc((pixels[i] >> GSHIFT) & 0xff, outfile);
				fputc((pixels[i] >> BSHIFT) & 0xff, outfile);
			}
			fflush(outfile);
			fclose(outfile);
		}

	}
};

class crayWorker: public UserTaskBase{
private:
	global_vars g_vars;
	sphere2_t* obj_array;
	vec3_t *lights;
	uint32_t *pixels;

	static thread_local int local_yres;
	static thread_local int local_startYresIndex;

	vec2_t urand[NRAN];
	int irand[NRAN];

	void render_scanline(int xsz, int ysz, int sl, uint32_t *fb, int samples);
	vec3_t trace(ray_t ray, int depth);
	int ray_sphere(const sphere2_t *sph, ray_t ray, spoint_t *sp);
	vec3_t shade( sphere2_t *obj, spoint_t *sp, int depth);
	ray_t get_primary_ray(int x, int y, int sample);
	vec3_t get_sample_pos(int x, int y, int sample);
	vec3_t jitter(int x, int y, int s);
	vec3_t reflect(vec3_t &v, vec3_t &n);
	vec3_t cross_product(vec3_t &v1, vec3_t &v2);


public:
	void initImpl(global_vars g_vars,
			sphere2_t* obj_array,
			uint32_t *pixels,
			vec3_t *lights);

	void runImpl(double runtime[][1]);

};



#endif /* TASK_H_ */
