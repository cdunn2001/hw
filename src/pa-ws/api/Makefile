all: schema.txt
	
headers = ObjectTypes.h ProcessStatusObject.h PostprimaryObject.h SocketObject.h StorageObject.h TransferObject.h

schema.txt: cpp2md.pl $(headers)
	perl ./cpp2md.pl $(headers) > $@
