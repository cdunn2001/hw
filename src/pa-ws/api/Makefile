all: schema.txt kes-paws-rest-api-swagger.yaml kes-paws-rest-api-swagger-temp.html
	
clean:
	rm -f schema.txt kes-paws-rest-api-swagger.yaml pa-ws.schemas.yaml kes-paws-rest-api-swagger-temp.html

distclean: clean
	rm -rf e2e_ve

headers = ObjectTypes.h PawsStatusObject.h ProcessStatusObject.h SocketObject.h PostprimaryObject.h StorageObject.h

schema.txt: cpp2md.pl $(headers)
	perl ./cpp2md.pl $(headers) > $@

pa-ws-schemas.yaml: cpp2oa.pl $(headers)
	perl ./cpp2oa.pl $(headers) > $@

kes-paws-rest-api-swagger.yaml: ../../../doc/kes-paws-rest-api.yaml pa-ws-schemas.yaml
	cat $^ > $@

kes-paws-rest-api-swagger-temp.html: kes-paws-rest-api-swagger.yaml
	./install_python3_ve.sh
	source e2e_ve/bin/activate && python swagger-yaml-to-html.py < $< > $@
